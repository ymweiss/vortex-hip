#!/bin/bash
# Test script for Vortex HIP Runtime

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Vortex HIP Runtime Test Script${NC}"
echo "=============================="
echo ""

# Check build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}ERROR: build directory not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

cd build

# Check library exists
if [ ! -f "libhip_vortex.so" ]; then
    echo -e "${RED}ERROR: libhip_vortex.so not found${NC}"
    echo "Please run ./build.sh first"
    exit 1
fi

# Check Vortex library path
if [ -z "$VORTEX_ROOT" ]; then
    VORTEX_ROOT=~/vortex
fi

VORTEX_LIB="$VORTEX_ROOT/build/runtime/libvortex.so"
if [ ! -f "$VORTEX_LIB" ]; then
    echo -e "${RED}ERROR: Vortex library not found at $VORTEX_LIB${NC}"
    exit 1
fi

# Set library path
export LD_LIBRARY_PATH=$(pwd):$VORTEX_ROOT/build/runtime:$LD_LIBRARY_PATH

echo "Testing Vortex HIP Runtime"
echo "Library path: $LD_LIBRARY_PATH"
echo ""

# Run tests
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -e "${YELLOW}Running test: $test_name${NC}"
    echo "Command: $test_cmd"
    echo ""

    if eval $test_cmd; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
    echo "---"
    echo ""
}

# Test 1: Vector Add Example
if [ -f "examples/vector_add" ]; then
    run_test "Vector Addition Example" "./examples/vector_add"
else
    echo -e "${YELLOW}Skipping vector_add (not built)${NC}"
    echo ""
fi

# Summary
echo "=============================="
echo "Test Results"
echo "=============================="
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
