#!/bin/bash
#
# compile_hip_v2.sh - Streamlined HIP to Vortex compilation pipeline
#
# This script compiles HIP source files using the new hipLaunchKernelGGL-based
# workflow. No Python preprocessing is required.
#
# Requirements:
#   - HIP source must use hipLaunchKernelGGL() for kernel launches
#   - Include "hip_runtime.h" for device code
#
# Pipeline:
#   1. cgeist: HIP → GPU dialect MLIR (hipLaunchKernelGGL expands to <<<>>>)
#   2. polygeist-opt: GPU MLIR → Vortex MLIR + host stubs (_args.h)
#   3. mlir-translate: MLIR → LLVM IR
#   4. llc/clang: LLVM IR → RISC-V binary (.vxbin)
#   5. g++: Host compilation with generated stubs
#
# Usage:
#   ./scripts/compile_hip_v2.sh <input.hip> [options]
#
# Options:
#   -o <output>     Output executable name (default: basename of input)
#   -k <name>       Kernel binary name (default: <kernel_name>.vxbin)
#   --device-only   Only compile device code (generates stubs and kernel)
#   --host-only     Only compile host code (requires existing stubs)
#   --keep-temps    Keep intermediate files
#   --verbose       Show all commands
#   --help          Show this help message
#
# See: Polygeist/docs/EMIT_VORTEX_WRAPPERS.md

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Tool paths
POLYGEIST_BUILD="${POLYGEIST_BUILD:-$REPO_ROOT/Polygeist/build}"
POLYGEIST_LLVM="${POLYGEIST_LLVM:-$REPO_ROOT/Polygeist/llvm-project/build}"
LLVM_VORTEX="${LLVM_VORTEX:-$REPO_ROOT/llvm-vortex/build}"
VORTEX_HOME="${VORTEX_HOME:-$REPO_ROOT/vortex}"

CGEIST="$POLYGEIST_BUILD/bin/cgeist"
POLYGEIST_OPT="$POLYGEIST_BUILD/bin/polygeist-opt"
MLIR_TRANSLATE="$POLYGEIST_LLVM/bin/mlir-translate"
CLANG_VORTEX="$LLVM_VORTEX/bin/clang"
VXBIN_PY="$VORTEX_HOME/kernel/scripts/vxbin.py"

RESOURCE_DIR="$REPO_ROOT/Polygeist/llvm-project/build/lib/clang/18"
LIBCXX_INCLUDE="$REPO_ROOT/Polygeist/llvm-project/libcxx/include"
HIP_DEVICE_INCLUDE="$REPO_ROOT/runtime/device"
HIP_HOST_INCLUDE="$REPO_ROOT/runtime/host"
HIP_RUNTIME_LIB="$REPO_ROOT/runtime/build"
# Device stub headers intercept problematic STL headers during device compilation
DEVICE_STUBS_INCLUDE="$POLYGEIST_BUILD/include/polygeist_device_stubs"

# Default options
OUTPUT=""
KERNEL_NAME=""
DEVICE_ONLY=0
HOST_ONLY=0
HOST_LIBRARY=0
KEEP_TEMPS=0
VERBOSE=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

run_cmd() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo "+ $@"
    fi
    "$@"
}

usage() {
    cat << 'EOF'
Streamlined HIP to Vortex Compilation Pipeline (v2)

This script uses the new hipLaunchKernelGGL-based workflow.
No Python preprocessing required - just use hipLaunchKernelGGL in your code.

Usage:
  ./scripts/compile_hip_v2.sh <input.hip> [options]

Options:
  -o <output>     Output executable name (default: basename of input)
  -k <name>       Kernel binary name (default: auto from kernel name)
  --device-only   Only compile device code
  --host-only     Only compile host code (requires existing stubs)
  --host-library  Compile launch wrappers as a shared library (experimental)
  --keep-temps    Keep intermediate files
  --verbose       Show all commands
  --help          Show this help

Example:
  ./scripts/compile_hip_v2.sh hip_tests/vecadd.hip -o vecadd

Source file requirements:
  - Use hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...)
  - Include "hip_runtime.h" (device header)

See: Polygeist/docs/EMIT_VORTEX_WRAPPERS.md
EOF
}

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUTPUT="$2"; shift 2 ;;
        -k) KERNEL_NAME="$2"; shift 2 ;;
        --device-only) DEVICE_ONLY=1; shift ;;
        --host-only) HOST_ONLY=1; shift ;;
        --host-library) HOST_LIBRARY=1; shift ;;
        --keep-temps) KEEP_TEMPS=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        --help) usage; exit 0 ;;
        -*) log_error "Unknown option: $1"; usage; exit 1 ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                log_error "Multiple input files not supported"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$INPUT_FILE" ]; then
    log_error "No input file specified"
    usage
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    log_error "Input file not found: $INPUT_FILE"
    exit 1
fi

# Derive names
BASENAME=$(basename "$INPUT_FILE" .hip)
BASENAME=$(basename "$BASENAME" .cu)
WORK_DIR=$(dirname "$INPUT_FILE")
OUTPUT="${OUTPUT:-$BASENAME}"

# Check tools
check_tool() {
    if [ ! -f "$1" ]; then
        log_error "$2 not found at: $1"
        exit 1
    fi
}

log_info "Checking tools..."
check_tool "$CGEIST" "cgeist"
check_tool "$POLYGEIST_OPT" "polygeist-opt"

# Check if libc++ is available (required for -stdlib=libc++ flag)
CLANG_PP="$POLYGEIST_LLVM/bin/clang++"
if [ -f "$CLANG_PP" ]; then
    if ! "$CLANG_PP" -stdlib=libc++ -x c++ -E - < /dev/null > /dev/null 2>&1; then
        log_warn "libc++ may not be available. Install with: apt install libc++-dev libc++abi-dev"
    fi
fi

# Temp files
TEMP_CU="/tmp/${BASENAME}_$$.cu"
GPU_MLIR="/tmp/${BASENAME}_$$.gpu.mlir"
VORTEX_MLIR="/tmp/${BASENAME}_$$.vortex.mlir"
KERNEL_LL="/tmp/${BASENAME}_$$.ll"

cleanup() {
    if [ "$KEEP_TEMPS" -eq 0 ]; then
        rm -f "$TEMP_CU" "$GPU_MLIR" "$VORTEX_MLIR" "$KERNEL_LL"
    else
        log_info "Keeping temp files:"
        [ -f "$GPU_MLIR" ] && echo "  $GPU_MLIR"
        [ -f "$VORTEX_MLIR" ] && echo "  $VORTEX_MLIR"
        [ -f "$KERNEL_LL" ] && echo "  $KERNEL_LL"
    fi
}
trap cleanup EXIT

#==============================================================================
# Stage 1: HIP Source Transformation + GPU MLIR Generation
#==============================================================================
compile_device_mlir() {
    log_info "Stage 1: HIP source transformation + GPU MLIR"

    # Copy to .cu for cgeist (it prefers .cu extension)
    if [[ "$INPUT_FILE" == *.hip ]]; then
        cp "$INPUT_FILE" "$TEMP_CU"
        CGEIST_INPUT="$TEMP_CU"
    else
        CGEIST_INPUT="$INPUT_FILE"
    fi

    # Stage 1a: Transform HIP source to insert kernel wrappers
    # This ensures kernel arguments maintain correct order during MLIR codegen
    # Also generates _args.h stub headers with correct argument order from AST
    TRANSFORMED_CU="/tmp/${BASENAME}_transformed_$$.cu"
    log_info "  Transforming HIP source (inserting kernel wrappers)..."
    run_cmd "$CGEIST" "$CGEIST_INPUT" \
        --transform-hip-source \
        --transform-only \
        --transform-output="$TRANSFORMED_CU" \
        --stub-output-dir="$WORK_DIR" \
        --cuda-lower \
        --emit-cuda \
        --cuda-gpu-arch=sm_60 \
        -nocudalib -nocudainc \
        -resource-dir="$RESOURCE_DIR" \
        -I"$DEVICE_STUBS_INCLUDE" \
        -I"$HIP_DEVICE_INCLUDE" \
        -I"$REPO_ROOT" \
        --function='*' \
        -S 2>&1

    if [ ! -f "$TRANSFORMED_CU" ]; then
        log_warn "Source transformation failed - using original source"
        TRANSFORMED_CU="$CGEIST_INPUT"
    else
        log_success "Source transformed (wrappers inserted)"
        if [ "$KEEP_TEMPS" -eq 1 ]; then
            cp "$TRANSFORMED_CU" "$WORK_DIR/${BASENAME}_transformed.cu"
            log_info "  Kept: $WORK_DIR/${BASENAME}_transformed.cu"
        fi
    fi

    # Stage 1b: Compile transformed source to GPU MLIR
    log_info "  Compiling to GPU dialect MLIR..."
    run_cmd "$CGEIST" "$TRANSFORMED_CU" \
        --cuda-lower \
        --emit-cuda \
        --use-original-gpu-block-size \
        --vortex-single-kernel \
        --cuda-gpu-arch=sm_60 \
        -nocudalib -nocudainc \
        -resource-dir="$RESOURCE_DIR" \
        -I"$DEVICE_STUBS_INCLUDE" \
        -I"$HIP_DEVICE_INCLUDE" \
        -I"$REPO_ROOT" \
        --function='*' \
        --output-intermediate-gpu=1 \
        --dump-hip-kernels \
        -S \
        -o "$GPU_MLIR" 2>&1

    # Save transformed source for host compilation (always needed)
    # The transformed source has the conditional wrapper that calls the generated stub
    if [ "$TRANSFORMED_CU" != "$CGEIST_INPUT" ]; then
        HOST_SOURCE="$WORK_DIR/${BASENAME}_transformed.cu"
        cp "$TRANSFORMED_CU" "$HOST_SOURCE"
        # Cleanup original temp file unless keeping temps
        if [ "$KEEP_TEMPS" -eq 0 ]; then
            rm -f "$TRANSFORMED_CU"
        fi
    else
        HOST_SOURCE="$CGEIST_INPUT"
    fi

    if [ ! -f "$GPU_MLIR" ]; then
        log_error "cgeist failed to generate GPU MLIR"
        exit 1
    fi

    # Verify gpu.launch was generated (from wrapper calls)
    if grep -q "gpu.launch\|gpu.launch_func" "$GPU_MLIR"; then
        log_success "GPU launch operations found"
    else
        log_warn "No gpu.launch found - ensure hipLaunchKernelGGL is used in source"
    fi
}

#==============================================================================
# Stage 2: GPU MLIR → Vortex MLIR + Host Stubs
#==============================================================================
convert_to_vortex() {
    log_info "Stage 2: GPU MLIR → Vortex MLIR (generating host stubs)"

    # Change to work directory so stubs are generated there
    pushd "$WORK_DIR" > /dev/null

    # Strip host-only functions before lowering to device code
    run_cmd "$POLYGEIST_OPT" "$GPU_MLIR" \
        --convert-gpu-to-vortex \
        --strip-host-only-functions \
        -o "$VORTEX_MLIR" 2>&1

    popd > /dev/null

    if [ ! -f "$VORTEX_MLIR" ]; then
        log_error "polygeist-opt failed to generate Vortex MLIR"
        exit 1
    fi

    # Check for generated stub files
    STUBS_GENERATED=$(ls "$WORK_DIR"/*_args.h 2>/dev/null | wc -l)
    if [ "$STUBS_GENERATED" -gt 0 ]; then
        log_success "Generated $STUBS_GENERATED host stub file(s)"
        ls "$WORK_DIR"/*_args.h 2>/dev/null | while read f; do
            echo "  $(basename $f)"
        done
    else
        log_warn "No host stubs generated - kernel outlining may not have occurred"
    fi
}

#==============================================================================
# Stage 3: Vortex MLIR → LLVM IR
#==============================================================================
mlir_to_llvm() {
    log_info "Stage 3: Vortex MLIR → LLVM IR"

    check_tool "$MLIR_TRANSLATE" "mlir-translate"

    MLIR_OPT="$POLYGEIST_LLVM/bin/mlir-opt"
    check_tool "$MLIR_OPT" "mlir-opt"

    # Stage 3a: Lower MLIR to LLVM dialect
    LLVM_DIALECT="/tmp/${BASENAME}_$$.llvm.mlir"
    log_info "  Lowering to LLVM dialect..."
    run_cmd "$MLIR_OPT" "$VORTEX_MLIR" \
        --convert-scf-to-cf \
        --convert-arith-to-llvm \
        --finalize-memref-to-llvm \
        --convert-index-to-llvm=index-bitwidth=32 \
        --convert-func-to-llvm \
        --convert-cf-to-llvm \
        --reconcile-unrealized-casts \
        -o "$LLVM_DIALECT" 2>&1

    # Stage 3b: Generate Vortex main wrapper
    MAIN_MLIR="/tmp/${BASENAME}_$$.main.mlir"
    log_info "  Generating Vortex main wrapper..."
    run_cmd "$POLYGEIST_OPT" "$LLVM_DIALECT" \
        --generate-vortex-main \
        -o "$MAIN_MLIR" 2>&1

    # Stage 3c: Translate to LLVM IR
    log_info "  Translating to LLVM IR..."
    run_cmd "$MLIR_TRANSLATE" \
        --mlir-to-llvmir \
        "$MAIN_MLIR" \
        -o "$KERNEL_LL" 2>&1

    # Cleanup intermediate files
    if [ "$KEEP_TEMPS" -eq 0 ]; then
        rm -f "$LLVM_DIALECT" "$MAIN_MLIR"
    else
        log_info "  Kept: $LLVM_DIALECT"
        log_info "  Kept: $MAIN_MLIR"
    fi

    if [ ! -f "$KERNEL_LL" ]; then
        log_error "mlir-translate failed"
        exit 1
    fi

    log_success "LLVM IR generated"
}

#==============================================================================
# Stage 4: LLVM IR → RISC-V Binary
#==============================================================================
compile_kernel_binary() {
    log_info "Stage 4: LLVM IR → RISC-V binary"

    LLC_VORTEX="$LLVM_VORTEX/bin/llc"
    check_tool "$LLC_VORTEX" "llc (llvm-vortex)"
    check_tool "$CLANG_VORTEX" "clang (llvm-vortex)"

    # Get toolchain paths
    LIBC_VORTEX="${LIBC_VORTEX:-$HOME/tools/libc32}"
    LIBCRT_VORTEX="${LIBCRT_VORTEX:-$HOME/tools/libcrt32}"
    RISCV_TOOLCHAIN="${RISCV_TOOLCHAIN:-$HOME/tools/riscv32-gnu-toolchain}"

    # Extract kernel name from generated metadata
    META_FILE=$(ls "$WORK_DIR"/*.meta.json 2>/dev/null | head -1)
    if [ -n "$META_FILE" ] && [ -f "$META_FILE" ]; then
        KERNEL_BASE=$(basename "$META_FILE" .meta.json)
        VXBIN_NAME="${KERNEL_NAME:-${KERNEL_BASE}.vxbin}"
    else
        VXBIN_NAME="${KERNEL_NAME:-${BASENAME}_kernel.vxbin}"
    fi

    KERNEL_OBJ="/tmp/${BASENAME}_$$.o"
    KERNEL_ELF="/tmp/${BASENAME}_$$.elf"

    # Stage 4a: LLVM IR → RISC-V object
    log_info "  Compiling LLVM IR to RISC-V object..."
    run_cmd "$LLC_VORTEX" \
        --mtriple=riscv32-unknown-unknown-elf \
        -march=riscv32 \
        -mattr=+m,+a,+f,+vortex \
        --vortex-branch-divergence=1 \
        -filetype=obj \
        "$KERNEL_LL" \
        -o "$KERNEL_OBJ" 2>&1

    if [ ! -f "$KERNEL_OBJ" ]; then
        log_error "LLC compilation failed"
        exit 1
    fi

    # Stage 4b: Link to ELF
    log_info "  Linking to ELF..."
    if [ -f "$VORTEX_HOME/build/kernel/libvortex.a" ] && [ -d "$LIBC_VORTEX" ]; then
        run_cmd "$CLANG_VORTEX" \
            -target riscv32-unknown-elf \
            -march=rv32imaf -mabi=ilp32f \
            -mcmodel=medany \
            -fno-rtti -fno-exceptions \
            -nostartfiles -nostdlib \
            "$KERNEL_OBJ" \
            -Wl,-Bstatic,--gc-sections,-z,norelro \
            -Wl,-T,"$VORTEX_HOME/kernel/scripts/link32.ld" \
            -Wl,--defsym=STARTUP_ADDR=0x80000000 \
            "$VORTEX_HOME/build/kernel/libvortex.a" \
            -L"$LIBC_VORTEX/lib" -lm -lc \
            "$LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a" \
            -o "$KERNEL_ELF" 2>&1 || {
                log_warn "ELF linking failed - toolchain may be missing"
                log_warn "Keeping object file instead"
                mv "$KERNEL_OBJ" "$WORK_DIR/${VXBIN_NAME%.vxbin}.o"
                return 0
            }
    else
        log_warn "Vortex libraries or libc not found - skipping ELF linking"
        mv "$KERNEL_OBJ" "$WORK_DIR/${VXBIN_NAME%.vxbin}.o"
        log_success "Kernel object: $WORK_DIR/${VXBIN_NAME%.vxbin}.o"
        return 0
    fi

    rm -f "$KERNEL_OBJ"

    # Stage 4c: ELF → Vortex binary
    log_info "  Converting to Vortex binary..."
    if [ -f "$VXBIN_PY" ]; then
        export OBJCOPY="$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-objcopy"
        run_cmd python3 "$VXBIN_PY" \
            "$KERNEL_ELF" \
            "$WORK_DIR/$VXBIN_NAME" 2>&1
        rm -f "$KERNEL_ELF"
    else
        # Fallback: keep ELF file
        mv "$KERNEL_ELF" "$WORK_DIR/$VXBIN_NAME"
        log_warn "vxbin.py not found - kernel saved as ELF"
    fi

    log_success "Kernel binary: $WORK_DIR/$VXBIN_NAME"
}

#==============================================================================
# Stage 5a: Host Library Compilation (Experimental)
# Compiles launch wrappers from MLIR to a shared library
#==============================================================================
compile_host_library() {
    log_info "Stage 5a: Host library compilation (experimental)"

    MLIR_OPT="$POLYGEIST_LLVM/bin/mlir-opt"
    check_tool "$MLIR_OPT" "mlir-opt"
    check_tool "$MLIR_TRANSLATE" "mlir-translate"

    # Convert gpu.launch_func to host runtime calls
    HOST_MLIR="/tmp/${BASENAME}_host_$$.mlir"
    log_info "  Converting launch operations to host calls..."
    run_cmd "$POLYGEIST_OPT" "$GPU_MLIR" \
        --convert-gpu-launch-to-host-call \
        -o "$HOST_MLIR" 2>&1

    if [ ! -f "$HOST_MLIR" ]; then
        log_error "Failed to convert launch operations"
        exit 1
    fi

    # Lower to LLVM dialect
    HOST_LLVM_MLIR="/tmp/${BASENAME}_host_llvm_$$.mlir"
    log_info "  Lowering to LLVM dialect..."
    run_cmd "$MLIR_OPT" "$HOST_MLIR" \
        --convert-scf-to-cf \
        --convert-arith-to-llvm \
        --finalize-memref-to-llvm \
        --convert-index-to-llvm \
        --convert-func-to-llvm \
        --convert-cf-to-llvm \
        --reconcile-unrealized-casts \
        -o "$HOST_LLVM_MLIR" 2>&1

    # Translate to LLVM IR
    HOST_LL="/tmp/${BASENAME}_host_$$.ll"
    log_info "  Translating to LLVM IR..."
    run_cmd "$MLIR_TRANSLATE" \
        --mlir-to-llvmir \
        "$HOST_LLVM_MLIR" \
        -o "$HOST_LL" 2>&1

    if [ ! -f "$HOST_LL" ]; then
        log_error "Failed to generate host LLVM IR"
        rm -f "$HOST_MLIR" "$HOST_LLVM_MLIR"
        exit 1
    fi

    # Compile to object file (for static linking into host executable)
    HOST_OBJ="$WORK_DIR/${BASENAME}_launch.o"
    log_info "  Compiling to object file..."

    # Use Polygeist's clang for host compilation
    HOST_CLANG="$POLYGEIST_LLVM/bin/clang"
    if [ ! -f "$HOST_CLANG" ]; then
        HOST_CLANG="clang"  # Fallback to system clang
    fi

    run_cmd "$HOST_CLANG" -c \
        -O2 \
        "$HOST_LL" \
        -o "$HOST_OBJ" 2>&1

    # Cleanup intermediate files
    if [ "$KEEP_TEMPS" -eq 0 ]; then
        rm -f "$HOST_MLIR" "$HOST_LLVM_MLIR" "$HOST_LL"
    else
        log_info "  Kept: $HOST_MLIR"
        log_info "  Kept: $HOST_LL"
    fi

    if [ -f "$HOST_OBJ" ]; then
        log_success "Host launch object: $HOST_OBJ"
    else
        log_error "Host object compilation failed"
        exit 1
    fi
}

#==============================================================================
# Stage 5b: Link Host Executable (with launch object)
#==============================================================================
link_host_executable() {
    log_info "Stage 5b: Linking host executable"

    HOST_OBJ="$WORK_DIR/${BASENAME}_launch.o"
    if [ ! -f "$HOST_OBJ" ]; then
        log_error "Host launch object not found: $HOST_OBJ"
        exit 1
    fi

    # Use Polygeist's clang for host compilation
    HOST_CLANG="$POLYGEIST_LLVM/bin/clang++"
    if [ ! -f "$HOST_CLANG" ]; then
        HOST_CLANG="g++"  # Fallback to system g++
    fi

    # Compile the original HIP source for host (excluding kernel code)
    # First compile source to object, then link with launch object
    HOST_MAIN_OBJ="/tmp/${BASENAME}_main_$$.o"

    log_info "  Compiling host source..."
    run_cmd "$HOST_CLANG" -std=c++17 -c \
        -x c++ \
        -D__HIP_PLATFORM_VORTEX__ \
        -DHIP_HOST_COMPILATION=1 \
        -I"$HIP_HOST_INCLUDE" \
        -I"$WORK_DIR" \
        -I"$REPO_ROOT" \
        "$INPUT_FILE" \
        -o "$HOST_MAIN_OBJ" 2>&1

    if [ ! -f "$HOST_MAIN_OBJ" ]; then
        log_error "Host source compilation failed"
        exit 1
    fi

    log_info "  Linking executable..."
    run_cmd "$HOST_CLANG" \
        -L"$HIP_RUNTIME_LIB" \
        -L"$VORTEX_HOME/build/runtime" \
        -Wl,-rpath,"$HIP_RUNTIME_LIB" \
        -Wl,-rpath,"$VORTEX_HOME/build/runtime" \
        "$HOST_MAIN_OBJ" \
        "$HOST_OBJ" \
        -lhip_vortex -lvortex \
        -o "$WORK_DIR/$OUTPUT" 2>&1

    rm -f "$HOST_MAIN_OBJ"

    if [ -f "$WORK_DIR/$OUTPUT" ]; then
        log_success "Host executable: $WORK_DIR/$OUTPUT"
    else
        log_error "Host linking failed"
        exit 1
    fi
}

#==============================================================================
# Stage 5: Host Compilation
#==============================================================================
compile_host() {
    log_info "Stage 5: Host compilation"

    # Use transformed source if available (has conditional wrapper for host/device)
    # HOST_SOURCE is set by compile_device_mlir
    COMPILE_SOURCE="${HOST_SOURCE:-$INPUT_FILE}"

    # Find generated stub files (for metadata)
    STUB_FILES=$(ls "$WORK_DIR"/*_args.h 2>/dev/null | tr '\n' ' ')

    if [ -z "$STUB_FILES" ]; then
        log_warn "No host stubs found - kernel launch may not work"
    fi

    # Compile transformed source with HIP_HOST_COMPILATION defined
    # This causes the wrapper to call the generated stub (vortexLaunchKernel)
    # instead of using <<<>>> syntax
    # Include paths:
    #   - HIP_HOST_INCLUDE/hip: hip_runtime.h with host-side definitions
    #   - HIP_HOST_INCLUDE: hip_vortex_runtime.h
    #   - WORK_DIR: generated *_args.h stubs
    run_cmd g++ -std=c++17 \
        -D__HIP_PLATFORM_VORTEX__ \
        -DHIP_HOST_COMPILATION=1 \
        -I"$HIP_HOST_INCLUDE/hip" \
        -I"$HIP_HOST_INCLUDE" \
        -I"$WORK_DIR" \
        -I"$REPO_ROOT" \
        -x c++ \
        "$COMPILE_SOURCE" \
        -L"$HIP_RUNTIME_LIB" \
        -L"$VORTEX_HOME/build/runtime" \
        -Wl,-rpath,"$HIP_RUNTIME_LIB" \
        -Wl,-rpath,"$VORTEX_HOME/build/runtime" \
        -lhip_vortex \
        -o "$WORK_DIR/$OUTPUT" 2>&1

    if [ -f "$WORK_DIR/$OUTPUT" ]; then
        log_success "Host executable: $WORK_DIR/$OUTPUT"
    else
        log_error "Host compilation failed"
        exit 1
    fi
}

#==============================================================================
# Main
#==============================================================================
echo ""
echo "=========================================="
echo "HIP to Vortex Compilation Pipeline (v2)"
echo "=========================================="
echo "Input: $INPUT_FILE"
echo ""

if [ "$HOST_ONLY" -eq 0 ]; then
    compile_device_mlir

    if [ "$HOST_LIBRARY" -eq 1 ]; then
        # Host library mode: compile device kernel + host launch object
        convert_to_vortex
        mlir_to_llvm
        compile_kernel_binary
        compile_host_library

        # Link host executable unless --device-only
        if [ "$DEVICE_ONLY" -eq 0 ]; then
            link_host_executable
        fi
    else
        # Standard mode: compile device kernel + generate stubs
        convert_to_vortex
        mlir_to_llvm
        compile_kernel_binary

        # Compile host with stubs unless --device-only
        if [ "$DEVICE_ONLY" -eq 0 ]; then
            compile_host
        fi
    fi
fi

echo ""
log_success "Build complete!"
