#!/bin/bash
#
# setup_dependencies.sh - Build all dependencies for vortex_hip on a fresh clone
#
# This script builds all required components in the correct order:
#   1. Vortex (runtime + toolchain)
#   2. Polygeist (HIP -> MLIR compiler)
#   3. llvm-vortex (LLVM IR -> RISC-V compiler)
#   4. HIP Runtime Library
#
# Usage:
#   ./scripts/setup_dependencies.sh [options]
#
# Options:
#   --skip-vortex       Skip Vortex build (if already built)
#   --skip-polygeist    Skip Polygeist build (if already built)
#   --skip-llvm-vortex  Skip llvm-vortex build (if already built)
#   --skip-runtime      Skip HIP runtime build (if already built)
#   --force-vortex      Force rebuild Vortex (delete existing build)
#   --force-polygeist   Force rebuild Polygeist (delete existing build)
#   --force-llvm-vortex Force rebuild llvm-vortex (delete existing build)
#   --force-runtime     Force rebuild HIP runtime (delete existing build)
#   --force-all         Force rebuild all components
#   --jobs <N>          Number of parallel jobs (default: auto-detect)
#   --tooldir <path>    Toolchain install directory (default: $HOME/tools)
#   --help              Show this help message
#
# Prerequisites:
#   - Linux (tested on Ubuntu 22.04, 24.04)
#   - GCC 11+ or Clang 14+
#   - CMake 3.20+
#   - Ninja (recommended) or Make
#   - Python 3.8+
#   - ~200 GB disk space
#
# System dependencies (installed automatically with sudo):
#   - build-essential, zlib1g-dev, libtinfo-dev, libncurses5
#   - uuid-dev, libboost-serialization-dev, libpng-dev, libhwloc-dev
#
# Note: System dependency installation is automatically skipped if sudo
# is unavailable or requires a password. Use --skip-system-deps to
# suppress the warning message.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default options
SKIP_VORTEX=0
SKIP_POLYGEIST=0
SKIP_LLVM_VORTEX=0
SKIP_RUNTIME=0
SKIP_SYSTEM_DEPS=0
FORCE_VORTEX=0
FORCE_POLYGEIST=0
FORCE_LLVM_VORTEX=0
FORCE_RUNTIME=0
JOBS=""
TOOLDIR="$HOME/tools"

# Logging functions
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

log_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Print usage
usage() {
    cat << 'EOF'
Build all dependencies for vortex_hip

Usage:
  ./scripts/setup_dependencies.sh [options]

Options:
  --skip-vortex       Skip Vortex build (if already built)
  --skip-polygeist    Skip Polygeist build (if already built)
  --skip-llvm-vortex  Skip llvm-vortex build (if already built)
  --skip-runtime      Skip HIP runtime build (if already built)
  --skip-system-deps  Skip system dependency installation (requires sudo)
  --force-vortex      Force rebuild Vortex (delete existing build)
  --force-polygeist   Force rebuild Polygeist (delete existing build)
  --force-llvm-vortex Force rebuild llvm-vortex (delete existing build)
  --force-runtime     Force rebuild HIP runtime (delete existing build)
  --force-all         Force rebuild all components
  --jobs <N>          Number of parallel jobs (default: auto-detect)
  --tooldir <path>    Toolchain install directory (default: $HOME/tools)
  --help              Show this help message

Build order:
  1. Vortex (runtime + RISC-V toolchain)
  2. Polygeist (LLVM/MLIR/Clang + Polygeist)
  3. llvm-vortex (LLVM with Vortex RISC-V extensions)
  4. HIP Runtime Library

Estimated time: 2-4 hours (depending on CPU cores)
Disk space required: ~200 GB

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-vortex)
            SKIP_VORTEX=1
            shift
            ;;
        --skip-polygeist)
            SKIP_POLYGEIST=1
            shift
            ;;
        --skip-llvm-vortex)
            SKIP_LLVM_VORTEX=1
            shift
            ;;
        --skip-runtime)
            SKIP_RUNTIME=1
            shift
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            shift
            ;;
        --force-vortex)
            FORCE_VORTEX=1
            shift
            ;;
        --force-polygeist)
            FORCE_POLYGEIST=1
            shift
            ;;
        --force-llvm-vortex)
            FORCE_LLVM_VORTEX=1
            shift
            ;;
        --force-runtime)
            FORCE_RUNTIME=1
            shift
            ;;
        --force-all)
            FORCE_VORTEX=1
            FORCE_POLYGEIST=1
            FORCE_LLVM_VORTEX=1
            FORCE_RUNTIME=1
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --tooldir)
            TOOLDIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Auto-detect number of jobs if not specified
if [ -z "$JOBS" ]; then
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    # Limit to avoid running out of memory during LLVM builds
    if [ "$JOBS" -gt 16 ]; then
        JOBS=16
    fi
fi

log_info "Using $JOBS parallel jobs"
log_info "Toolchain directory: $TOOLDIR"
log_info "Project root: $PROJECT_ROOT"

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites"

    local missing=()

    # Check for required commands
    command -v cmake >/dev/null 2>&1 || missing+=("cmake")
    command -v ninja >/dev/null 2>&1 || missing+=("ninja-build")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v g++ >/dev/null 2>&1 || missing+=("g++")
    command -v git >/dev/null 2>&1 || missing+=("git")

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_info "Install with: sudo apt-get install ${missing[*]}"
        exit 1
    fi

    # Check CMake version
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 20 ]); then
        log_error "CMake 3.20+ required, found $CMAKE_VERSION"
        exit 1
    fi

    # Check GCC version
    GCC_VERSION=$(g++ -dumpversion | cut -d. -f1)
    if [ "$GCC_VERSION" -lt 11 ]; then
        log_warn "GCC 11+ recommended, found GCC $GCC_VERSION"
        log_info "Consider: sudo apt-get install gcc-11 g++-11"
    fi

    log_success "Prerequisites check passed"
}

# Initialize submodules
init_submodules() {
    log_step "Initializing git submodules"

    cd "$PROJECT_ROOT"

    if [ ! -f "vortex/README.md" ] || [ ! -f "Polygeist/README.md" ] || [ ! -f "llvm-vortex/README.md" ]; then
        log_info "Initializing submodules (this may take a while)..."
        git submodule update --init --recursive
    else
        log_info "Submodules already initialized"
    fi

    log_success "Submodules ready"
}

# Install system dependencies
install_system_deps() {
    if [ "$SKIP_SYSTEM_DEPS" -eq 1 ]; then
        log_info "Skipping system dependency installation (--skip-system-deps)"
        return 0
    fi

    log_step "Installing system dependencies"

    # Check if sudo is available and working
    if ! command -v sudo >/dev/null 2>&1; then
        log_warn "sudo not found, skipping system dependency installation"
        log_info "Please install dependencies manually if needed"
        return 0
    fi

    # Test if sudo works (non-interactive)
    if ! sudo -n true 2>/dev/null; then
        log_warn "sudo requires password or is not configured, skipping system dependency installation"
        log_info "Run with sudo permissions or use --skip-system-deps to suppress this warning"
        return 0
    fi

    if command -v apt-get >/dev/null 2>&1; then
        log_info "Detected Debian/Ubuntu system"

        # Check if we need sudo
        if [ -f "$PROJECT_ROOT/vortex/ci/install_dependencies.sh" ]; then
            log_info "Running Vortex dependency installer..."
            sudo "$PROJECT_ROOT/vortex/ci/install_dependencies.sh" || {
                log_warn "Vortex dependency installer failed, trying manual install..."
                sudo apt-get update
                sudo apt-get install -y \
                    build-essential \
                    zlib1g-dev \
                    libtinfo-dev \
                    libncurses5 \
                    uuid-dev \
                    libboost-serialization-dev \
                    libpng-dev \
                    libhwloc-dev \
                    gcc-11 g++-11
            }
        else
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                zlib1g-dev \
                libtinfo-dev \
                libncurses5 \
                uuid-dev \
                libboost-serialization-dev \
                libpng-dev \
                libhwloc-dev
        fi

        log_success "System dependencies installed"
    elif command -v yum >/dev/null 2>&1; then
        log_info "Detected RHEL/CentOS system"
        sudo yum install -y \
            libpng-devel \
            boost boost-devel boost-serialization \
            libuuid-devel \
            hwloc hwloc-devel \
            gmp-devel
        log_success "System dependencies installed"
    else
        log_warn "Unknown package manager, please install dependencies manually"
        log_info "Required: build-essential, zlib1g-dev, libtinfo-dev, uuid-dev, libboost-serialization-dev, libpng-dev, libhwloc-dev"
    fi
}

# Build Vortex
build_vortex() {
    if [ "$SKIP_VORTEX" -eq 1 ]; then
        log_info "Skipping Vortex build (--skip-vortex)"
        return 0
    fi

    log_step "Step 1/4: Building Vortex"

    cd "$PROJECT_ROOT/vortex"

    # Force rebuild if requested
    if [ "$FORCE_VORTEX" -eq 1 ] && [ -d "build" ]; then
        log_warn "Force rebuild requested, removing existing Vortex build..."
        rm -rf build
    fi

    # Check if already built
    if [ -f "build/runtime/libvortex.so" ] && [ -f "build/kernel/libvortex.a" ]; then
        log_info "Vortex already built, skipping (use --force-vortex to rebuild)"
        return 0
    fi

    # Configure build
    log_info "Configuring Vortex build..."
    mkdir -p build && cd build
    ../configure --xlen=32 --tooldir="$TOOLDIR"

    # Install toolchain
    log_info "Installing RISC-V toolchain (this may take 30-60 minutes)..."
    ./ci/toolchain_install.sh --all

    # Source environment
    log_info "Setting up toolchain environment..."
    source ./ci/toolchain_env.sh

    # Build Vortex
    log_info "Building Vortex runtime and kernel libraries..."
    make -s -j"$JOBS"

    # Verify
    if [ -f "runtime/libvortex.so" ] && [ -f "kernel/libvortex.a" ]; then
        log_success "Vortex built successfully"
    else
        log_error "Vortex build failed"
        exit 1
    fi
}

# Build Polygeist
build_polygeist() {
    if [ "$SKIP_POLYGEIST" -eq 1 ]; then
        log_info "Skipping Polygeist build (--skip-polygeist)"
        return 0
    fi

    log_step "Step 2/4: Building Polygeist"

    cd "$PROJECT_ROOT/Polygeist"

    # Force rebuild if requested
    if [ "$FORCE_POLYGEIST" -eq 1 ]; then
        if [ -d "build" ]; then
            log_warn "Force rebuild requested, removing existing Polygeist build..."
            rm -rf build
        fi
        if [ -d "llvm-project/build" ]; then
            log_warn "Force rebuild requested, removing existing LLVM/MLIR build..."
            rm -rf llvm-project/build
        fi
    fi

    # Check if already built
    if [ -f "build/bin/cgeist" ] && [ -f "build/bin/polygeist-opt" ]; then
        log_info "Polygeist already built, skipping (use --force-polygeist to rebuild)"
        return 0
    fi

    # Build LLVM/MLIR/Clang first
    log_info "Building LLVM/MLIR/Clang (this may take 30-60 minutes)..."
    mkdir -p llvm-project/build && cd llvm-project/build

    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON

    ninja -j"$JOBS"

    # Verify LLVM build
    if [ ! -f "bin/mlir-opt" ]; then
        log_error "LLVM/MLIR build failed"
        exit 1
    fi
    log_success "LLVM/MLIR/Clang built"

    # Build Polygeist
    log_info "Building Polygeist..."
    cd "$PROJECT_ROOT/Polygeist"
    mkdir -p build && cd build

    cmake -G Ninja .. \
        -DMLIR_DIR="$PROJECT_ROOT/Polygeist/llvm-project/build/lib/cmake/mlir" \
        -DCLANG_DIR="$PROJECT_ROOT/Polygeist/llvm-project/build/lib/cmake/clang" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
        -DCMAKE_BUILD_TYPE=Release

    ninja cgeist polygeist-opt -j"$JOBS"

    # Verify
    if [ -f "bin/cgeist" ] && [ -f "bin/polygeist-opt" ]; then
        log_success "Polygeist built successfully"
        ./bin/cgeist --version || true
    else
        log_error "Polygeist build failed"
        exit 1
    fi
}

# Build llvm-vortex
build_llvm_vortex() {
    if [ "$SKIP_LLVM_VORTEX" -eq 1 ]; then
        log_info "Skipping llvm-vortex build (--skip-llvm-vortex)"
        return 0
    fi

    log_step "Step 3/4: Building llvm-vortex"

    cd "$PROJECT_ROOT/llvm-vortex"

    # Force rebuild if requested
    if [ "$FORCE_LLVM_VORTEX" -eq 1 ] && [ -d "build" ]; then
        log_warn "Force rebuild requested, removing existing llvm-vortex build..."
        rm -rf build
    fi

    # Check if already built
    if [ -f "build/bin/clang" ] && [ -f "build/bin/llc" ]; then
        log_info "llvm-vortex already built, skipping (use --force-llvm-vortex to rebuild)"
        return 0
    fi

    log_info "Building llvm-vortex (this may take 30-60 minutes)..."
    mkdir -p build && cd build

    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_TARGETS_TO_BUILD="RISCV" \
        -DCMAKE_BUILD_TYPE=Release

    ninja clang llc -j"$JOBS"

    # Verify
    if [ -f "bin/clang" ] && [ -f "bin/llc" ]; then
        log_success "llvm-vortex built successfully"
        ./bin/llc --version | grep -i riscv || log_warn "RISC-V target may not be enabled"
    else
        log_error "llvm-vortex build failed"
        exit 1
    fi
}

# Build HIP runtime
build_hip_runtime() {
    if [ "$SKIP_RUNTIME" -eq 1 ]; then
        log_info "Skipping HIP runtime build (--skip-runtime)"
        return 0
    fi

    log_step "Step 4/4: Building HIP Runtime Library"

    cd "$PROJECT_ROOT/runtime"

    # Force rebuild if requested
    if [ "$FORCE_RUNTIME" -eq 1 ] && [ -d "build" ]; then
        log_warn "Force rebuild requested, removing existing HIP runtime build..."
        rm -rf build
    fi

    # Check if already built
    if [ -f "build/libhip_vortex.so" ]; then
        log_info "HIP runtime already built, skipping (use --force-runtime to rebuild)"
        return 0
    fi

    log_info "Building HIP runtime library..."
    mkdir -p build && cd build

    cmake .. \
        -DVORTEX_ROOT="$PROJECT_ROOT/vortex" \
        -DBUILD_EXAMPLES=OFF

    make -j"$JOBS"

    # Verify
    if [ -f "libhip_vortex.so" ] || [ -f "libhip_vortex.so.1.0.0" ]; then
        log_success "HIP runtime built successfully"
    else
        log_error "HIP runtime build failed"
        exit 1
    fi
}

# Print final summary
print_summary() {
    log_step "Build Complete!"

    echo "All components built successfully."
    echo ""
    echo "To use the toolchain, run:"
    echo "  export VORTEX_HIP_HOME=$PROJECT_ROOT"
    echo "  source \$VORTEX_HIP_HOME/vortex/build/ci/toolchain_env.sh"
    echo ""
    echo "To compile a HIP program:"
    echo "  ./scripts/compile_hip.sh hip_tests/vecadd.hip"
    echo ""
    echo "To run on the Vortex simulator:"
    echo "  export VORTEX_DRIVER=simx"
    echo "  export LD_LIBRARY_PATH=\$VORTEX_HIP_HOME/vortex/build/runtime:\$VORTEX_HIP_HOME/runtime/build:\$LD_LIBRARY_PATH"
    echo "  ./vecadd"
    echo ""

    # Show build locations
    echo "Build artifacts:"
    echo "  Vortex:       $PROJECT_ROOT/vortex/build/"
    echo "  Polygeist:    $PROJECT_ROOT/Polygeist/build/"
    echo "  llvm-vortex:  $PROJECT_ROOT/llvm-vortex/build/"
    echo "  HIP runtime:  $PROJECT_ROOT/runtime/build/"
    echo "  Toolchain:    $TOOLDIR/"
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo " vortex_hip Dependency Setup Script"
    echo "=========================================="
    echo ""

    check_prerequisites
    init_submodules
    install_system_deps
    build_vortex
    build_polygeist
    build_llvm_vortex
    build_hip_runtime
    print_summary
}

# Run main
main
