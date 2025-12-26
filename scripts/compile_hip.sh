#!/bin/bash
#
# compile_hip.sh - Full HIP to Vortex compilation pipeline
#
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  NOTE: This script uses the legacy Python-based workflow.                  ║
# ║                                                                            ║
# ║  The preferred approach is to use hipLaunchKernelGGL() in your HIP code,   ║
# ║  which enables direct compilation with cgeist without Python preprocessing.║
# ║  ConvertGPUToVortex pass now generates complete host stubs automatically.  ║
# ║                                                                            ║
# ║  See: Polygeist/docs/EMIT_VORTEX_WRAPPERS.md                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# This script compiles a HIP source file through the entire pipeline:
#   1. Source transformation (inject_kernel_launchers.py) [LEGACY]
#   2. Device compilation (Polygeist → MLIR → LLVM IR → RISC-V)
#   3. Host stub generation (generate_host_stubs.py) [LEGACY - now done by pass]
#   4. Host compilation (g++)
#
# Usage:
#   ./scripts/compile_hip.sh <input.hip> [options]
#
# Options:
#   -o <output>     Output executable name (default: basename of input)
#   -k <name>       Kernel binary name (default: kernel.vxbin)
#   --host-only     Only compile host code (requires existing stubs)
#   --device-only   Only compile device code (generates stubs and kernel)
#   --keep-temps    Keep intermediate files
#   --verbose       Show all commands
#   --help          Show this help message
#
# Environment variables:
#   VORTEX_HIP_HOME   Path to vortex_hip repository (default: script directory/..)
#   VORTEX_HOME       Path to vortex source directory (default: $VORTEX_HIP_HOME/vortex)
#   LLVM_VORTEX       Path to llvm-vortex build (default: $VORTEX_HIP_HOME/llvm-vortex/build)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default paths
# VORTEX_HIP_HOME: Root of vortex_hip repository
VORTEX_HIP_HOME="${VORTEX_HIP_HOME:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# VORTEX_HOME: Root of vortex submodule (source directory, contains kernel/ and build/)
VORTEX_HOME="${VORTEX_HOME:-$VORTEX_HIP_HOME/vortex}"
# LLVM_VORTEX: llvm-vortex build directory
LLVM_VORTEX="${LLVM_VORTEX:-$VORTEX_HIP_HOME/llvm-vortex/build}"
# POLYGEIST_BUILD: Polygeist build directory
POLYGEIST_BUILD="${POLYGEIST_BUILD:-$VORTEX_HIP_HOME/Polygeist/build}"
# POLYGEIST_LLVM: Polygeist's LLVM build directory
POLYGEIST_LLVM="${POLYGEIST_LLVM:-$VORTEX_HIP_HOME/Polygeist/llvm-project/build}"

# Tools
CGEIST="$POLYGEIST_BUILD/bin/cgeist"
POLYGEIST_OPT="$POLYGEIST_BUILD/bin/polygeist-opt"
MLIR_OPT="$POLYGEIST_LLVM/bin/mlir-opt"
MLIR_TRANSLATE="$POLYGEIST_LLVM/bin/mlir-translate"
CLANG_VORTEX="$LLVM_VORTEX/bin/clang"
VXBIN_PY="$VORTEX_HOME/kernel/scripts/vxbin.py"

# Default options
OUTPUT=""
KERNEL_NAME="kernel.vxbin"
MLIR_INPUT=""
HOST_ONLY=0
DEVICE_ONLY=0
KEEP_TEMPS=0
VERBOSE=0

# Print usage
usage() {
    cat << 'EOF'
HIP to Vortex Compilation Pipeline

This script compiles a HIP source file through the entire pipeline:
  1. Source transformation (inject_kernel_launchers.py)
  2. Device compilation (Polygeist → MLIR → LLVM IR → RISC-V)
  3. Host stub generation (generate_host_stubs.py)
  4. Host compilation (g++)

Usage:
  ./scripts/compile_hip.sh <input.hip> [options]

Options:
  -o <output>     Output executable name (default: basename of input)
  -k <name>       Kernel binary name (default: kernel.vxbin)
  --mlir <file>   Use existing GPU dialect MLIR file (skip cgeist step)
  --host-only     Only compile host code (requires existing stubs)
  --device-only   Only compile device code (generates stubs and kernel)
  --keep-temps    Keep intermediate files
  --verbose       Show all commands
  --help          Show this help message

Environment variables:
  VORTEX_HIP_HOME   Path to vortex_hip repository (default: script directory/..)
  VORTEX_HOME       Path to vortex source directory (default: $VORTEX_HIP_HOME/vortex)
  LLVM_VORTEX       Path to llvm-vortex build (default: $VORTEX_HIP_HOME/llvm-vortex/build)

Examples:
  # Compile a HIP kernel
  ./scripts/compile_hip.sh vecadd.hip

  # Compile with custom output names
  ./scripts/compile_hip.sh vecadd.hip -o vecadd_app -k vecadd.vxbin

  # Keep intermediate files for debugging
  ./scripts/compile_hip.sh vecadd.hip --keep-temps --verbose
EOF
    exit 0
}

# Print colored message
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Run command with optional verbose output
run_cmd() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo -e "${YELLOW}+ $*${NC}"
    fi
    "$@"
}

# Check if a tool exists
check_tool() {
    local tool="$1"
    local name="$2"
    if [ ! -x "$tool" ]; then
        log_error "$name not found at: $tool"
        log_error "Please build $name or set the appropriate environment variable"
        exit 1
    fi
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o)
                OUTPUT="$2"
                shift 2
                ;;
            -k)
                KERNEL_NAME="$2"
                shift 2
                ;;
            --mlir)
                MLIR_INPUT="$2"
                shift 2
                ;;
            --host-only)
                HOST_ONLY=1
                shift
                ;;
            --device-only)
                DEVICE_ONLY=1
                shift
                ;;
            --keep-temps)
                KEEP_TEMPS=1
                shift
                ;;
            --verbose|-v)
                VERBOSE=1
                shift
                ;;
            --help|-h)
                usage
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                ;;
            *)
                if [ -z "$INPUT" ]; then
                    INPUT="$1"
                else
                    log_error "Multiple input files not supported"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [ -z "$INPUT" ]; then
        log_error "No input file specified"
        usage
    fi

    if [ ! -f "$INPUT" ]; then
        log_error "Input file not found: $INPUT"
        exit 1
    fi

    # Set default output name
    if [ -z "$OUTPUT" ]; then
        OUTPUT="$(basename "${INPUT%.*}")"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ "$DEVICE_ONLY" -eq 0 ] || [ "$HOST_ONLY" -eq 0 ]; then
        # Need Polygeist for device compilation
        check_tool "$CGEIST" "cgeist (Polygeist)"
        check_tool "$POLYGEIST_OPT" "polygeist-opt"
        check_tool "$MLIR_OPT" "mlir-opt"
        check_tool "$MLIR_TRANSLATE" "mlir-translate"
    fi

    if [ "$HOST_ONLY" -eq 0 ]; then
        # Need llvm-vortex for kernel compilation
        check_tool "$CLANG_VORTEX" "clang (llvm-vortex)"

        # Check for Vortex kernel library
        if [ ! -f "$VORTEX_HOME/build/kernel/libvortex.a" ]; then
            log_warn "Vortex kernel library not found at $VORTEX_HOME/build/kernel/libvortex.a"
            log_warn "Kernel linking may fail"
        fi
    fi

    # Check for HIP runtime library (required for host linking)
    if [ "$DEVICE_ONLY" -eq 0 ]; then
        HIP_RUNTIME_LIB_PATH="$VORTEX_HIP_HOME/runtime/build/libhip_vortex.so"
        if [ ! -f "$HIP_RUNTIME_LIB_PATH" ]; then
            log_error "HIP runtime library not found at: $HIP_RUNTIME_LIB_PATH"
            log_error "Build it with:"
            log_error "  cd $VORTEX_HIP_HOME/runtime && mkdir -p build && cd build && cmake .. && make"
            exit 1
        fi
    fi

    # Check Python scripts
    if [ ! -f "$SCRIPT_DIR/polygeist/inject_kernel_launchers.py" ]; then
        log_error "inject_kernel_launchers.py not found"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/generate_host_stubs.py" ]; then
        log_error "generate_host_stubs.py not found"
        exit 1
    fi

    log_success "Prerequisites OK"
}

# Create temp directory
setup_temp_dir() {
    if [ "$KEEP_TEMPS" -eq 1 ]; then
        TEMP_DIR="./build_${OUTPUT}"
        mkdir -p "$TEMP_DIR"
    else
        TEMP_DIR=$(mktemp -d)
        trap "rm -rf $TEMP_DIR" EXIT
    fi
    log_info "Working directory: $TEMP_DIR"
}

# Step 1: Transform source (for host compilation only)
transform_source() {
    log_info "Step 1: Preparing source for host compilation..."

    # Run inject_kernel_launchers.py to prepare source for host compilation
    # This script:
    # - Adds conditional includes using __CUDA__ macro
    # - Wraps kernel definitions with #ifdef __CUDA__
    # - Generates extern declarations for host compilation
    # - Injects synthetic launch wrappers required by Polygeist
    #
    # Note: For device compilation, hip-to-gpu-dialect.sh handles the
    # source transformation internally.
    log_info "  Running inject_kernel_launchers.py"
    run_cmd python3 "$SCRIPT_DIR/polygeist/inject_kernel_launchers.py" \
        "$INPUT" \
        "$TEMP_DIR/transformed.cu"

    log_success "Source transformed: $TEMP_DIR/transformed.cu"
}

# Step 2: Device compilation
compile_device() {
    log_info "Step 2: Compiling device code..."

    # 2a: HIP → GPU MLIR
    log_info "  2a: HIP → GPU Dialect MLIR"

    # Check if pre-existing kernel MLIR exists (from previous successful compilation)
    BASENAME=$(basename "${INPUT%.*}")
    PRE_EXISTING_MLIR="$VORTEX_HIP_HOME/hip_tests/mlir_output/${BASENAME}_kernel.mlir"

    if [ -n "$MLIR_INPUT" ] && [ -f "$MLIR_INPUT" ]; then
        # User provided an MLIR file directly
        log_info "  Using provided MLIR file: $MLIR_INPUT"
        cp "$MLIR_INPUT" "$TEMP_DIR/gpu.mlir"
    else
        # Try to generate using hip-to-gpu-dialect.sh
        # Use the transformed source (with host code gated) instead of original
        HIP_TO_GPU_SCRIPT="$SCRIPT_DIR/polygeist/hip-to-gpu-dialect.sh"
        MLIR_GENERATED=0
        if [ -f "$HIP_TO_GPU_SCRIPT" ]; then
            log_info "  Running hip-to-gpu-dialect.sh on transformed source..."
            if run_cmd "$HIP_TO_GPU_SCRIPT" "$TEMP_DIR/transformed.cu" "$TEMP_DIR/gpu.mlir" 2>/dev/null; then
                log_success "  Generated GPU MLIR"
                MLIR_GENERATED=1
            else
                log_warn "  hip-to-gpu-dialect.sh failed, checking for cached MLIR..."
            fi
        fi

        # Fallback to cached MLIR only if generation failed
        if [ "$MLIR_GENERATED" -eq 0 ]; then
            if [ -f "$PRE_EXISTING_MLIR" ]; then
                log_warn "  Using cached MLIR as fallback: $PRE_EXISTING_MLIR"
                cp "$PRE_EXISTING_MLIR" "$TEMP_DIR/gpu.mlir"
            else
                log_error "MLIR generation failed and no cached MLIR found"
                log_error "Expected cache location: $PRE_EXISTING_MLIR"
                exit 1
            fi
        fi
    fi

    # 2b: GPU MLIR → Vortex MLIR (+ metadata)
    log_info "  2b: GPU Dialect → Vortex MLIR (+ metadata generation)"
    pushd "$TEMP_DIR" > /dev/null
    run_cmd "$POLYGEIST_OPT" gpu.mlir \
        --convert-gpu-to-vortex \
        -o vortex.mlir
    popd > /dev/null

    # 2c: Generate host stubs from metadata and extract kernel name
    log_info "  2c: Generating host stubs from metadata"
    META_FILES=$(find "$TEMP_DIR" -name "*.meta.json" 2>/dev/null || true)
    if [ -z "$META_FILES" ]; then
        log_warn "No metadata files generated - stubs may be incomplete"
    else
        run_cmd python3 "$SCRIPT_DIR/generate_host_stubs.py" \
            $META_FILES \
            -o "$TEMP_DIR/kernel_stubs.h"
        log_success "Generated: $TEMP_DIR/kernel_stubs.h"

        # Extract kernel name from metadata for .vxbin filename
        # Only update KERNEL_NAME if it's still the default
        if [ "$KERNEL_NAME" = "kernel.vxbin" ]; then
            # Find the primary kernel metadata (prefer __polygeist_launch_ prefix)
            PRIMARY_META=$(echo "$META_FILES" | tr ' ' '\n' | grep "__polygeist_launch_" | head -1)
            if [ -n "$PRIMARY_META" ]; then
                # Extract kernel name from JSON: look for "kernel_name" field
                EXTRACTED_NAME=$(python3 -c "import json; f=open('$PRIMARY_META'); d=json.load(f); name=d.get('kernel_name',''); name=name.replace('__polygeist_launch_',''); import re; name=re.sub(r'_*\d{6,}$','',name); print(name+'_kernel' if name and not name.endswith('_kernel') else name)" 2>/dev/null || true)
                if [ -n "$EXTRACTED_NAME" ]; then
                    KERNEL_NAME="${EXTRACTED_NAME}.vxbin"
                    log_info "  Using kernel name from metadata: $KERNEL_NAME"
                fi
            fi
        fi
    fi

    # 2d: MLIR lowering → LLVM Dialect
    log_info "  2d: MLIR lowering → LLVM Dialect"
    # Note: index-bitwidth=32 for RV32 Vortex target
    run_cmd "$MLIR_OPT" "$TEMP_DIR/vortex.mlir" \
        --convert-scf-to-cf \
        --convert-arith-to-llvm \
        --finalize-memref-to-llvm \
        --convert-index-to-llvm=index-bitwidth=32 \
        --convert-func-to-llvm \
        --convert-cf-to-llvm \
        --reconcile-unrealized-casts \
        -o "$TEMP_DIR/llvm_dialect.mlir"

    # 2e: Generate Vortex main wrapper
    log_info "  2e: Generating Vortex main wrapper"
    run_cmd "$POLYGEIST_OPT" "$TEMP_DIR/llvm_dialect.mlir" \
        --generate-vortex-main \
        -o "$TEMP_DIR/with_main.mlir"

    # 2f: MLIR → LLVM IR
    log_info "  2f: MLIR → LLVM IR"
    run_cmd "$MLIR_TRANSLATE" \
        --mlir-to-llvmir \
        "$TEMP_DIR/with_main.mlir" \
        -o "$TEMP_DIR/kernel.ll"

    # 2g: LLVM IR → RISC-V ELF
    log_info "  2g: LLVM IR → RISC-V ELF"

    # Get toolchain paths from environment or use defaults
    # These follow the Vortex naming convention: libc32, libcrt32 for rv32
    LIBC_VORTEX="${LIBC_VORTEX:-$HOME/tools/libc32}"
    LIBCRT_VORTEX="${LIBCRT_VORTEX:-$HOME/tools/libcrt32}"
    RISCV_TOOLCHAIN="${RISCV_TOOLCHAIN:-$HOME/tools/riscv32-gnu-toolchain}"
    LLC_VORTEX="$LLVM_VORTEX/bin/llc"

    # Step 1: Compile LLVM IR to object file using llc
    # This allows us to set the correct RISC-V target triple
    # (mlir-translate outputs x86_64 target triple by default)
    # Note: +vortex enables Vortex-specific ISA extensions
    # --vortex-branch-divergence=1 enables automatic split/join insertion for divergent branches
    log_info "    Compiling LLVM IR to RISC-V object"
    run_cmd "$LLC_VORTEX" \
        --mtriple=riscv32-unknown-unknown-elf \
        -march=riscv32 \
        -mattr=+m,+a,+f,+vortex \
        --vortex-branch-divergence=1 \
        -filetype=obj \
        "$TEMP_DIR/kernel.ll" \
        -o "$TEMP_DIR/kernel.o" || {
            log_warn "LLC compilation failed"
            log_warn "Continuing with LLVM IR only..."
            return 0
        }

    # Step 2: Link object file with Vortex runtime
    # The vx_get_* accessor functions are in libvortex.a (vx_spawn.c)
    # Note: -z norelro is needed to avoid .got section placement issues
    log_info "    Linking kernel ELF"
    run_cmd "$CLANG_VORTEX" \
        -target riscv32-unknown-elf \
        -march=rv32imaf -mabi=ilp32f \
        -mcmodel=medany \
        -fno-rtti -fno-exceptions \
        -nostartfiles -nostdlib \
        "$TEMP_DIR/kernel.o" \
        -Wl,-Bstatic,--gc-sections,-z,norelro \
        -Wl,-T,"$VORTEX_HOME/kernel/scripts/link32.ld" \
        -Wl,--defsym=STARTUP_ADDR=0x80000000 \
        "$VORTEX_HOME/build/kernel/libvortex.a" \
        -L"$LIBC_VORTEX/lib" -lm -lc \
        "$LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a" \
        -o "$TEMP_DIR/kernel.elf" || {
            log_warn "Kernel ELF linking failed - this may be due to missing toolchain"
            log_warn "Continuing with LLVM IR only..."
            return 0
        }

    # 2h: ELF → Vortex binary
    log_info "  2h: ELF → Vortex binary format"
    if [ -f "$VXBIN_PY" ] && [ -f "$TEMP_DIR/kernel.elf" ]; then
        # vxbin.py needs RISC-V objcopy to read the ELF
        export OBJCOPY="$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-objcopy"
        run_cmd python3 "$VXBIN_PY" "$TEMP_DIR/kernel.elf" "$TEMP_DIR/$KERNEL_NAME"
        log_success "Generated: $TEMP_DIR/$KERNEL_NAME"
    else
        log_warn "vxbin.py not found or kernel.elf missing - skipping vxbin generation"
    fi

    log_success "Device compilation complete"
}

# Step 3: Host compilation
compile_host() {
    log_info "Step 3: Compiling host code..."

    # Check if stubs exist
    if [ ! -f "$TEMP_DIR/kernel_stubs.h" ]; then
        log_warn "kernel_stubs.h not found - creating empty stubs"
        echo "// Empty stubs - no metadata available" > "$TEMP_DIR/kernel_stubs.h"
    fi

    # Create a simple host wrapper if none exists
    HOST_SOURCE="$TEMP_DIR/transformed.cu"

    # Compile host code
    # Note: -x c++ is needed because g++ doesn't recognize .cu as C++ source
    log_info "  Compiling host object"
    run_cmd g++ -x c++ -c "$HOST_SOURCE" \
        -o "$TEMP_DIR/host.o" \
        -I"$VORTEX_HIP_HOME/runtime/host" \
        -I"$TEMP_DIR" \
        -std=c++17 \
        -DVORTEX_HIP_HOST \
        || {
            log_warn "Host compilation with transformed source failed"
            log_warn "You may need to compile host code separately"
            return 0
        }

    # Link host executable
    # Links against:
    # - libhip_vortex.so: HIP API implementation (hipMalloc, hipMemcpy, etc.)
    # - libvortex.so: Low-level Vortex runtime (vx_dev_open, vx_start, etc.)
    HIP_RUNTIME_LIB="$VORTEX_HIP_HOME/runtime/build"
    log_info "  Linking host executable"
    run_cmd g++ "$TEMP_DIR/host.o" \
        -o "$OUTPUT" \
        -L"$HIP_RUNTIME_LIB" \
        -L"$VORTEX_HOME/build/runtime" \
        -lhip_vortex -lvortex \
        -Wl,-rpath,"$HIP_RUNTIME_LIB" \
        -Wl,-rpath,"$VORTEX_HOME/build/runtime" \
        || {
            log_error "Host linking failed"
            log_error "Ensure libhip_vortex.so is built: cd runtime/build && cmake .. && make"
            log_error "Ensure libvortex.so is built: cd vortex/build && make"
            exit 1
        }

    log_success "Host compilation complete: $OUTPUT"
}

# Copy output files
copy_outputs() {
    log_info "Copying output files..."

    # Copy kernel binary
    if [ -f "$TEMP_DIR/$KERNEL_NAME" ]; then
        cp "$TEMP_DIR/$KERNEL_NAME" "./$KERNEL_NAME"
        log_success "Kernel binary: ./$KERNEL_NAME"
    fi

    # Copy stubs header
    if [ -f "$TEMP_DIR/kernel_stubs.h" ]; then
        cp "$TEMP_DIR/kernel_stubs.h" "./kernel_stubs.h"
        log_success "Host stubs: ./kernel_stubs.h"
    fi

    # Copy LLVM IR for debugging
    if [ "$KEEP_TEMPS" -eq 1 ] && [ -f "$TEMP_DIR/kernel.ll" ]; then
        cp "$TEMP_DIR/kernel.ll" "./${OUTPUT}_kernel.ll"
        log_info "LLVM IR: ./${OUTPUT}_kernel.ll"
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "Compilation Summary"
    echo "========================================"
    echo "Input:        $INPUT"

    if [ -f "./$OUTPUT" ]; then
        echo "Executable:   ./$OUTPUT"
    fi

    if [ -f "./$KERNEL_NAME" ]; then
        echo "Kernel:       ./$KERNEL_NAME"
    fi

    if [ -f "./kernel_stubs.h" ]; then
        echo "Stubs:        ./kernel_stubs.h"
    fi

    if [ "$KEEP_TEMPS" -eq 1 ]; then
        echo "Temp files:   $TEMP_DIR/"
    fi

    echo ""
    echo "To run on simulator:"
    echo "  export VORTEX_DRIVER=simx"
    echo "  export LD_LIBRARY_PATH=$VORTEX_HOME/build/runtime:\$LD_LIBRARY_PATH"
    echo "  ./$OUTPUT"
    echo ""
}

# Main
main() {
    echo "========================================"
    echo "HIP to Vortex Compilation Pipeline"
    echo "========================================"
    echo ""

    parse_args "$@"
    check_prerequisites
    setup_temp_dir

    if [ "$HOST_ONLY" -eq 0 ]; then
        transform_source
        compile_device
    fi

    if [ "$DEVICE_ONLY" -eq 0 ]; then
        compile_host
    fi

    copy_outputs
    print_summary

    log_success "Done!"
}

main "$@"
