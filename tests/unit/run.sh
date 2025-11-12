#!/bin/bash
# Unit test runner for Vortex HIP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "==================================================================="
echo "  Vortex HIP Unit Tests"
echo "==================================================================="
echo ""

# Create build directory
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring tests with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build tests
echo ""
echo "Building tests..."
make -j$(nproc)

# Run tests
echo ""
echo "==================================================================="
echo "  Running Tests"
echo "==================================================================="
echo ""

# Fix library path issues (prefer system libraries over conda)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ctest --output-on-failure --verbose

# Summary
echo ""
echo "==================================================================="
echo "  Test Summary"
echo "==================================================================="
ctest --output-on-failure | grep "tests passed"

echo ""
echo "✅ All tests completed"
