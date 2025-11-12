#!/bin/bash
# Comprehensive test suite for basic_test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Basic Test - Comprehensive Evaluation"
echo "========================================"
echo ""

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo ">>> $name"
    if ./run.sh "$@" 2>&1 | grep -q "✅ Test PASSED"; then
        echo "    ✅ PASSED"
        ((PASS++))
    else
        echo "    ❌ FAILED"
        ((FAIL++))
    fi
    echo ""
}

# Test 0: Memory operations
run_test "Test 0: Memory copy (16 elements)" -t 0 -n 16
run_test "Test 0: Memory copy (256 elements)" -t 0 -n 256
run_test "Test 0: Memory copy (1024 elements)" -t 0 -n 1024

# Test 1: Kernel execution - single block
run_test "Test 1: Single block, 16 threads" -t 1 -n 16 -b 1
run_test "Test 1: Single block, 64 threads" -t 1 -n 64 -b 1
run_test "Test 1: Single block, 128 threads" -t 1 -n 128 -b 1
run_test "Test 1: Single block, 256 threads" -t 1 -n 256 -b 1

# Test 1: Kernel execution - multiple blocks
run_test "Test 1: 2 blocks × 32 threads (64 total)" -t 1 -n 32 -b 2
run_test "Test 1: 4 blocks × 64 threads (256 total)" -t 1 -n 64 -b 4
run_test "Test 1: 8 blocks × 32 threads (256 total)" -t 1 -n 32 -b 8
run_test "Test 1: 16 blocks × 16 threads (256 total)" -t 1 -n 16 -b 16

echo "========================================"
echo "Summary: $PASS passed, $FAIL failed"
echo "========================================"

exit $FAIL
