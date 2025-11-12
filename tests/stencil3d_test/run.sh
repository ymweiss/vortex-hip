#!/bin/bash
# Run stencil3d_test with proper Vortex environment
# This script sets up library paths and environment for Vortex simulators

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HIP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VORTEX_ROOT="$VORTEX_HIP_ROOT/vortex"

# Check if Vortex is built
if [ ! -d "$VORTEX_ROOT/build/runtime" ]; then
    echo "Error: Vortex not built. Please build Vortex first:"
    echo "  cd $VORTEX_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  ../configure --xlen=32"
    echo "  source ./ci/toolchain_env.sh"
    echo "  make -j\$(nproc)"
    exit 1
fi

# Check if HIP runtime is built
if [ ! -f "$VORTEX_HIP_ROOT/runtime/build/libhip_vortex.so" ]; then
    echo "Error: HIP runtime not built. Please build it first:"
    echo "  cd $VORTEX_HIP_ROOT/runtime"
    echo "  ./build.sh"
    exit 1
fi

# Check if test is built
if [ ! -f "$SCRIPT_DIR/stencil3d_test" ]; then
    echo "Error: Test not built. Building now..."
    cd "$SCRIPT_DIR"
    make clean && make
fi

# Set up environment
export VORTEX_HOME="$VORTEX_ROOT"
export LD_LIBRARY_PATH="$VORTEX_ROOT/build/runtime:$VORTEX_HIP_ROOT/runtime/build:$LD_LIBRARY_PATH"

# Select simulator (default: simx - fastest)
export VORTEX_DRIVER="${VORTEX_DRIVER:-simx}"

echo "=========================================="
echo "Vortex HIP stencil3d Test"
echo "=========================================="
echo "VORTEX_HOME: $VORTEX_HOME"
echo "VORTEX_DRIVER: $VORTEX_DRIVER"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Parse arguments - forward all args to stencil3d_test
echo "Running: ./stencil3d_test $@"
echo "=========================================="
echo ""

# Run the test
cd "$SCRIPT_DIR"
./stencil3d_test "$@"

RESULT=$?

echo ""
echo "=========================================="
if [ $RESULT -eq 0 ]; then
    echo "✅ Test PASSED"
else
    echo "❌ Test FAILED (exit code: $RESULT)"
fi
echo "=========================================="

exit $RESULT
