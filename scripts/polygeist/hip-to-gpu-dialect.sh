#!/bin/bash
# Script to convert HIP/CUDA code to GPU dialect IR using Polygeist

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input.cu/.hip> [output.mlir]"
    echo "Example: $0 hip_tests/simple_kernel.hip gpu_dialect.mlir"
    exit 1
fi

INPUT_FILE="$1"
# Default output file: replace .cu or .hip extension with .mlir
if [ -z "$2" ]; then
    OUTPUT_FILE="${INPUT_FILE%.cu}.mlir"
    OUTPUT_FILE="${OUTPUT_FILE%.hip}.mlir"
else
    OUTPUT_FILE="$2"
fi

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POLYGEIST_SRC="$REPO_ROOT/Polygeist"
CGEIST="$POLYGEIST_SRC/build/bin/cgeist"
RESOURCE_DIR="$POLYGEIST_SRC/llvm-project/build/lib/clang/18"
HIP_INCLUDE="$REPO_ROOT/hip/include"

if [ ! -f "$CGEIST" ]; then
    echo "Error: cgeist not found at $CGEIST"
    echo "Please build Polygeist first using scripts/polygeist/build-polygeist.sh"
    exit 1
fi

echo "Converting $INPUT_FILE to GPU dialect IR..."
echo "Output: $OUTPUT_FILE"
echo "Using HIP headers: $HIP_INCLUDE"
echo "Using resource dir: $RESOURCE_DIR"

# Polygeist doesn't handle .hip extension properly, convert to .cu temporarily
TEMP_CU_FILE=""
if [[ "$INPUT_FILE" == *.hip ]]; then
    TEMP_CU_FILE="/tmp/polygeist_temp_$(basename "$INPUT_FILE" .hip).cu"
    cp "$INPUT_FILE" "$TEMP_CU_FILE"
    CGEIST_INPUT="$TEMP_CU_FILE"
    echo "Note: Converted .hip to temporary .cu file for Polygeist compatibility"
else
    CGEIST_INPUT="$INPUT_FILE"
fi

$CGEIST "$CGEIST_INPUT" \
    --cuda-lower \
    --emit-cuda \
    --cuda-gpu-arch=sm_60 \
    -nocudalib \
    -nocudainc \
    -resource-dir="$RESOURCE_DIR" \
    -I"$HIP_INCLUDE" \
    -I"$REPO_ROOT" \
    --function='*' \
    --output-intermediate-gpu=1 \
    -S \
    -o "$OUTPUT_FILE" 2>&1

CGEIST_EXIT=$?

# Clean up temporary file
if [ -n "$TEMP_CU_FILE" ] && [ -f "$TEMP_CU_FILE" ]; then
    rm "$TEMP_CU_FILE"
fi

if [ $CGEIST_EXIT -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "GPU dialect IR generated successfully!"
    echo ""

    # Filter kernel variants (keep only first variant per kernel)
    TEMP_OUTPUT="${OUTPUT_FILE}.unfiltered"
    mv "$OUTPUT_FILE" "$TEMP_OUTPUT"

    FILTER_SCRIPT="$SCRIPT_DIR/filter-gpu-variants.py"
    if [ -f "$FILTER_SCRIPT" ]; then
        echo "Filtering kernel variants..."
        if python3 "$FILTER_SCRIPT" "$TEMP_OUTPUT" "$OUTPUT_FILE" > /dev/null 2>&1; then
            rm "$TEMP_OUTPUT"
            echo "✓ Filtered to single variant per kernel"
        else
            # Filter failed, keep unfiltered version
            mv "$TEMP_OUTPUT" "$OUTPUT_FILE"
            echo "⚠ Warning: Variant filtering failed, keeping all variants"
        fi
    else
        # No filter script, keep unfiltered
        mv "$TEMP_OUTPUT" "$OUTPUT_FILE"
    fi

    echo ""
    echo "Verifying GPU dialect presence..."
    if grep -q "gpu.module\|gpu.func\|gpu.thread_id\|gpu.block_id" "$OUTPUT_FILE"; then
        echo "✓ GPU dialect operations found"
    else
        echo "⚠ Warning: No GPU dialect operations found in output"
    fi
else
    echo "Error: cgeist failed to generate output"
    echo "Check input file and Polygeist build"
    exit 1
fi
echo ""
echo "To view the output:"
echo "  cat $OUTPUT_FILE"
