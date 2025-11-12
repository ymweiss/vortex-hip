#!/bin/bash
# Quick test script for metadata generation

set -e

echo "==================================================================="
echo "  HIP Metadata Generation Test (Phase 1 - No Compiler Changes)"
echo "==================================================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ -z "$VORTEX_HOME" ]; then
    echo "Error: VORTEX_HOME not set"
    echo "Please run: export VORTEX_HOME=$HOME/vortex"
    exit 1
fi

if [ ! -d "$VORTEX_HOME" ]; then
    echo "Error: VORTEX_HOME directory not found: $VORTEX_HOME"
    exit 1
fi

if [ -z "$LLVM_VORTEX" ]; then
    echo "Error: LLVM_VORTEX not set"
    echo "Please run: source $VORTEX_HOME/ci/toolchain_env.sh"
    exit 1
fi

VORTEX_HIP_HOME="${VORTEX_HIP_HOME:-$HOME/vortex_hip}"

if [ ! -f "$VORTEX_HIP_HOME/runtime/build/libhip_vortex.so" ]; then
    echo "Error: HIP runtime not built"
    echo "Please run:"
    echo "  cd $VORTEX_HIP_HOME/runtime"
    echo "  ./build.sh"
    exit 1
fi

echo "✓ Prerequisites OK"
echo ""

# Show configuration
echo "Configuration:"
echo "  VORTEX_HOME:     $VORTEX_HOME"
echo "  VORTEX_HIP_HOME: $VORTEX_HIP_HOME"
echo "  LLVM_VORTEX:     $LLVM_VORTEX"
echo "  XLEN:            ${XLEN:-64}"
echo ""

# Clean previous build
echo "Cleaning previous build..."
make clean
echo ""

# Build
echo "==================================================================="
echo "  Building (this will show all 6 phases)"
echo "==================================================================="
echo ""
make all
echo ""

# Run test
echo "==================================================================="
echo "  Running Test"
echo "==================================================================="
echo ""
./vecadd_test 256
