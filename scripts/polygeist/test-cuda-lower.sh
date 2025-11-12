#!/bin/bash
set -e

CGEIST=/home/yaakov/vortex_hip/Polygeist/build/bin/cgeist
MLIR_OPT=/home/yaakov/vortex_hip/Polygeist/llvm-project/build/bin/mlir-opt
TEST_DIR=/home/yaakov/vortex_hip/Polygeist/cuda-lower-investigation
RESULTS_FILE=/home/yaakov/vortex_hip/Polygeist/cuda-lower-results.md

mkdir -p $TEST_DIR
cd $TEST_DIR

echo "# CUDA Lower Investigation" | tee $RESULTS_FILE
echo "Testing Polygeist's \`--cuda-lower\` flag to understand Phase 2 requirements" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE
echo "Date: $(date)" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Test 1: Use existing Polygeist CUDA test
echo "## Test 1: Existing Polygeist CUDA Test" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

CUDA_TEST=/home/yaakov/vortex_hip/Polygeist/tools/cgeist/Test/CUDA/polybench-cuda/gemver/gemver.cu

if [ -f "$CUDA_TEST" ]; then
    echo "Using test: \`$CUDA_TEST\`" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    echo "### Kernel Source (kernel_A):" | tee -a $RESULTS_FILE
    echo '```cpp' | tee -a $RESULTS_FILE
    sed -n '23,31p' $CUDA_TEST | tee -a $RESULTS_FILE
    echo '```' | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    # Try to compile with --cuda-lower
    echo "### Attempting compilation with --cuda-lower:" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    $CGEIST $CUDA_TEST --function=kernel_A -S --cuda-lower --nocudainc --nocudalib \
        > kernel_A_cuda_lower.mlir 2>&1

    COMPILE_RESULT=$?

    if [ $COMPILE_RESULT -eq 0 ]; then
        echo "✅ **SUCCESS**: Compiled with --cuda-lower" | tee -a $RESULTS_FILE
        echo "" | tee -a $RESULTS_FILE

        # Analyze output
        echo "### Output Analysis:" | tee -a $RESULTS_FILE
        echo "" | tee -a $RESULTS_FILE

        # Check for GPU dialect operations
        if grep -q "gpu\." kernel_A_cuda_lower.mlir; then
            echo "✅ **GPU dialect operations found**" | tee -a $RESULTS_FILE
            echo "" | tee -a $RESULTS_FILE
            echo "GPU operations:" | tee -a $RESULTS_FILE
            echo '```mlir' | tee -a $RESULTS_FILE
            grep "gpu\." kernel_A_cuda_lower.mlir | head -20 | tee -a $RESULTS_FILE
            echo '```' | tee -a $RESULTS_FILE
        else
            echo "⚠️ No GPU dialect operations found" | tee -a $RESULTS_FILE
        fi
        echo "" | tee -a $RESULTS_FILE

        # Check for threadIdx/blockIdx handling
        if grep -qi "threadIdx\|blockIdx" kernel_A_cuda_lower.mlir; then
            echo "✅ **threadIdx/blockIdx references found**" | tee -a $RESULTS_FILE
            echo "" | tee -a $RESULTS_FILE
            echo "Thread index handling:" | tee -a $RESULTS_FILE
            echo '```mlir' | tee -a $RESULTS_FILE
            grep -i "threadIdx\|blockIdx\|blockDim" kernel_A_cuda_lower.mlir | head -10 | tee -a $RESULTS_FILE
            echo '```' | tee -a $RESULTS_FILE
        else
            echo "ℹ️ threadIdx/blockIdx converted to other representation" | tee -a $RESULTS_FILE
        fi
        echo "" | tee -a $RESULTS_FILE

        # Show first 80 lines of output
        echo "### First 80 lines of MLIR output:" | tee -a $RESULTS_FILE
        echo '```mlir' | tee -a $RESULTS_FILE
        head -80 kernel_A_cuda_lower.mlir | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
        echo "" | tee -a $RESULTS_FILE

        # List all dialects used
        echo "### Dialects used:" | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
        grep -oh "[a-z_]*\." kernel_A_cuda_lower.mlir | sort -u | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE

    else
        echo "❌ **FAILED**: Could not compile with --cuda-lower" | tee -a $RESULTS_FILE
        echo "" | tee -a $RESULTS_FILE
        echo "### Error output:" | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
        head -50 kernel_A_cuda_lower.mlir | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
    fi

else
    echo "❌ Test file not found: $CUDA_TEST" | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Test 2: Simple synthetic CUDA kernel
echo "## Test 2: Simple Synthetic CUDA Kernel" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Copy the minimal cuda.h header
cp /home/yaakov/vortex_hip/Polygeist/tools/cgeist/Test/Verification/Inputs/cuda.h ./

cat > simple_cuda.cu <<'EOF'
#include "cuda.h"

__global__ void simple_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
EOF

echo "### Source:" | tee -a $RESULTS_FILE
echo '```cpp' | tee -a $RESULTS_FILE
cat simple_cuda.cu | tee -a $RESULTS_FILE
echo '```' | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

echo "### Compilation attempt:" | tee -a $RESULTS_FILE
$CGEIST simple_cuda.cu --function=simple_kernel -S --cuda-lower --nocudainc --nocudalib \
    > simple_cuda_lower.mlir 2>&1

SIMPLE_RESULT=$?

if [ $SIMPLE_RESULT -eq 0 ]; then
    echo "✅ **SUCCESS**: Simple CUDA kernel compiled" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    echo "### Output (first 100 lines):" | tee -a $RESULTS_FILE
    echo '```mlir' | tee -a $RESULTS_FILE
    head -100 simple_cuda_lower.mlir | tee -a $RESULTS_FILE
    echo '```' | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    # Check what happened to thread indexing
    echo "### Thread indexing analysis:" | tee -a $RESULTS_FILE
    if grep -q "gpu.thread_id" simple_cuda_lower.mlir; then
        echo "✅ Converted to \`gpu.thread_id\`" | tee -a $RESULTS_FILE
        echo '```mlir' | tee -a $RESULTS_FILE
        grep "gpu.thread_id\|gpu.block_id\|gpu.block_dim" simple_cuda_lower.mlir | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
    elif grep -qi "threadIdx" simple_cuda_lower.mlir; then
        echo "ℹ️ Kept as threadIdx variable" | tee -a $RESULTS_FILE
    else
        echo "⚠️ Different representation - investigating..." | tee -a $RESULTS_FILE
        echo '```mlir' | tee -a $RESULTS_FILE
        grep -A5 "blockIdx\|threadIdx\|idx.*=" simple_cuda_lower.mlir | head -20 | tee -a $RESULTS_FILE
        echo '```' | tee -a $RESULTS_FILE
    fi
else
    echo "❌ **FAILED**: Simple kernel compilation failed" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    echo "### Error:" | tee -a $RESULTS_FILE
    echo '```' | tee -a $RESULTS_FILE
    cat simple_cuda_lower.mlir | tee -a $RESULTS_FILE
    echo '```' | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Summary
echo "## Summary & Findings" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

if [ $COMPILE_RESULT -eq 0 ] || [ $SIMPLE_RESULT -eq 0 ]; then
    echo "### Key Discoveries:" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    echo "1. **\`--cuda-lower\` flag**: $([ $COMPILE_RESULT -eq 0 ] && echo '✅ Works' || echo '⚠️ Needs investigation')" | tee -a $RESULTS_FILE
    echo "2. **GPU dialect generation**: Check output files above" | tee -a $RESULTS_FILE
    echo "3. **Thread index handling**: See analysis sections" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    echo "### Files Generated:" | tee -a $RESULTS_FILE
    echo "- \`kernel_A_cuda_lower.mlir\` - Polybench CUDA test output" | tee -a $RESULTS_FILE
    echo "- \`simple_cuda_lower.mlir\` - Simple kernel output" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE

    echo "### Next Steps for Phase 2:" | tee -a $RESULTS_FILE
    echo "1. Analyze how \`threadIdx\`/\`blockIdx\` are represented" | tee -a $RESULTS_FILE
    echo "2. Understand GPU dialect operations used" | tee -a $RESULTS_FILE
    echo "3. Map GPU operations to Vortex equivalents" | tee -a $RESULTS_FILE
    echo "4. Test with more complex kernels" | tee -a $RESULTS_FILE
else
    echo "⚠️ **Both tests failed** - need to resolve compilation issues" | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    echo "Possible issues:" | tee -a $RESULTS_FILE
    echo "- CUDA headers needed" | tee -a $RESULTS_FILE
    echo "- Additional flags required" | tee -a $RESULTS_FILE
    echo "- Polygeist configuration" | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "---" | tee -a $RESULTS_FILE
echo "Full results: \`$RESULTS_FILE\`" | tee -a $RESULTS_FILE
echo "Test directory: \`$TEST_DIR\`" | tee -a $RESULTS_FILE
