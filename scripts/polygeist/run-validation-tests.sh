#!/bin/bash
set -e

CGEIST=/home/yaakov/vortex_hip/Polygeist/build/bin/cgeist
TEST_DIR=/home/yaakov/vortex_hip/Polygeist/validation-tests
RESULTS_FILE=/home/yaakov/vortex_hip/Polygeist/validation-results.txt

mkdir -p $TEST_DIR
cd $TEST_DIR

echo "=== Polygeist Validation Tests ===" | tee $RESULTS_FILE
echo "Date: $(date)" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Test 1: Simple function to SCF
echo "Test 1: Simple C function → MLIR" | tee -a $RESULTS_FILE
cat > test1_simple.c <<'EOF'
int add(int a, int b) {
    return a + b;
}
EOF

$CGEIST test1_simple.c --function=add -S > test1_output.mlir 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PASS: Simple function compiled" | tee -a $RESULTS_FILE
    echo "  Output:" | tee -a $RESULTS_FILE
    head -20 test1_output.mlir | sed 's/^/    /' | tee -a $RESULTS_FILE
else
    echo "  ✗ FAIL: Simple function failed" | tee -a $RESULTS_FILE
fi
echo "" | tee -a $RESULTS_FILE

# Test 2: Loop to scf.for
echo "Test 2: C loop → SCF dialect" | tee -a $RESULTS_FILE
cat > test2_loop.c <<'EOF'
void loop_sum(int n, int *arr) {
    for (int i = 0; i < n; i++) {
        arr[i] = i * 2;
    }
}
EOF

$CGEIST test2_loop.c --function=loop_sum -S > test2_output.mlir 2>&1
if [ $? -eq 0 ] && grep -q "scf.for" test2_output.mlir; then
    echo "  ✓ PASS: Loop converted to scf.for" | tee -a $RESULTS_FILE
    echo "  SCF operations found:" | tee -a $RESULTS_FILE
    grep "scf\." test2_output.mlir | head -5 | sed 's/^/    /' | tee -a $RESULTS_FILE
else
    echo "  ✗ FAIL: Loop conversion failed or no scf.for found" | tee -a $RESULTS_FILE
fi
echo "" | tee -a $RESULTS_FILE

# Test 3: Nested loops
echo "Test 3: Nested loops → Nested SCF" | tee -a $RESULTS_FILE
cat > test3_nested.c <<'EOF'
void matrix_init(int n, int m, int **matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = i + j;
        }
    }
}
EOF

$CGEIST test3_nested.c --function=matrix_init -S > test3_output.mlir 2>&1
if [ $? -eq 0 ]; then
    SCF_COUNT=$(grep -c "scf.for" test3_output.mlir || true)
    echo "  ✓ PASS: Nested loops compiled ($SCF_COUNT scf.for operations)" | tee -a $RESULTS_FILE
else
    echo "  ✗ FAIL: Nested loop conversion failed" | tee -a $RESULTS_FILE
fi
echo "" | tee -a $RESULTS_FILE

# Test 4: Conditional (if/else)
echo "Test 4: Conditional → SCF dialect" | tee -a $RESULTS_FILE
cat > test4_conditional.c <<'EOF'
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
EOF

$CGEIST test4_conditional.c --function=max -S > test4_output.mlir 2>&1
if [ $? -eq 0 ] && grep -q "scf.if" test4_output.mlir; then
    echo "  ✓ PASS: Conditional converted to scf.if" | tee -a $RESULTS_FILE
else
    echo "  ✗ FAIL: Conditional conversion failed" | tee -a $RESULTS_FILE
fi
echo "" | tee -a $RESULTS_FILE

# Test 5: Check if CUDA test infrastructure exists
echo "Test 5: CUDA infrastructure check" | tee -a $RESULTS_FILE
if [ -d "/home/yaakov/vortex_hip/Polygeist/tools/cgeist/Test/CUDA" ]; then
    CUDA_TESTS=$(find /home/yaakov/vortex_hip/Polygeist/tools/cgeist/Test/CUDA -name "*.cu" | wc -l)
    echo "  ✓ PASS: Found $CUDA_TESTS CUDA test files" | tee -a $RESULTS_FILE
else
    echo "  ⚠  SKIP: CUDA test directory not found" | tee -a $RESULTS_FILE
fi
echo "" | tee -a $RESULTS_FILE

echo "=== Summary ===" | tee -a $RESULTS_FILE
PASS_COUNT=$(grep -c "✓ PASS" $RESULTS_FILE || true)
FAIL_COUNT=$(grep -c "✗ FAIL" $RESULTS_FILE || true)
echo "Passed: $PASS_COUNT" | tee -a $RESULTS_FILE
echo "Failed: $FAIL_COUNT" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓✓✓ All validation tests passed!" | tee -a $RESULTS_FILE
else
    echo "⚠️  Some tests failed. Review results above." | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "Full results saved to: $RESULTS_FILE"
