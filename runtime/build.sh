#!/bin/bash
# Quick build script for Vortex HIP Runtime

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Vortex HIP Runtime Build Script${NC}"
echo "================================"
echo ""

# Check VORTEX_ROOT
if [ -z "$VORTEX_ROOT" ]; then
    VORTEX_ROOT=~/vortex
    echo -e "${YELLOW}VORTEX_ROOT not set, using default: $VORTEX_ROOT${NC}"
fi

# Verify Vortex installation
echo "Checking Vortex installation..."
if [ ! -f "$VORTEX_ROOT/runtime/include/vortex.h" ]; then
    echo -e "${RED}ERROR: Vortex headers not found at $VORTEX_ROOT/runtime/include/${NC}"
    echo "Please ensure VORTEX_ROOT is set correctly or Vortex is installed at ~/vortex"
    exit 1
fi

if [ ! -f "$VORTEX_ROOT/build/runtime/libvortex.so" ]; then
    echo -e "${RED}ERROR: Vortex library not found at $VORTEX_ROOT/build/runtime/libvortex.so${NC}"
    echo ""
    echo "Vortex must be built before building the HIP runtime."
    echo "Building Vortex requires multiple steps and installing dependencies."
    echo ""
    echo "Please build Vortex first by following these steps:"
    echo ""
    echo "1. Install dependencies (Ubuntu/Debian):"
    echo "   sudo apt-get install build-essential cmake git"
    echo "   # Plus Vortex-specific dependencies (see Vortex documentation)"
    echo ""
    echo "2. Configure and build Vortex:"
    echo "   cd $VORTEX_ROOT"
    echo "   mkdir -p build && cd build"
    echo "   ../configure"
    echo "   make -j\$(nproc)"
    echo ""
    echo "3. Verify the build produced the library:"
    echo "   ls -lh $VORTEX_ROOT/build/runtime/libvortex.so"
    echo ""
    echo "4. Then re-run this script:"
    echo "   cd $(dirname $(readlink -f $0))"
    echo "   ./build.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Vortex found at $VORTEX_ROOT"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure
echo "Configuring with CMake..."
cmake .. \
    -DVORTEX_ROOT=$VORTEX_ROOT \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: CMake configuration failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Configuration complete"
echo ""

# Build
echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Build complete"
echo ""

# Summary
echo "================================"
echo -e "${GREEN}Build Successful!${NC}"
echo ""
echo "Library: $(pwd)/libhip_vortex.so"
echo "Examples: $(pwd)/examples/"
echo ""
echo "To run the example:"
echo "  export LD_LIBRARY_PATH=$VORTEX_ROOT/build/runtime:\$LD_LIBRARY_PATH"
echo "  ./examples/vector_add"
echo ""
echo "Or run the test script:"
echo "  ./test.sh"
echo ""
