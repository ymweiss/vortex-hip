#!/bin/bash
# Batch convert all HIP tests to GPU dialect MLIR using Polygeist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HIP_TESTS_DIR="$REPO_ROOT/hip_tests"
OUTPUT_DIR="$HIP_TESTS_DIR/mlir_output"
CONVERT_SCRIPT="$SCRIPT_DIR/hip-to-gpu-dialect.sh"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Batch Converting HIP Tests to GPU Dialect"
echo "=========================================="
echo ""

# Find all .hip files
HIP_FILES=$(find "$HIP_TESTS_DIR" -maxdepth 1 -name "*.hip" -type f | sort)

TOTAL=0
SUCCESS=0
FAILED=0
FAILED_FILES=()

for HIP_FILE in $HIP_FILES; do
    BASENAME=$(basename "$HIP_FILE" .hip)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.mlir"

    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] Converting: $BASENAME.hip"
    echo "    Output: $OUTPUT_FILE"

    # Run conversion script
    if "$CONVERT_SCRIPT" "$HIP_FILE" "$OUTPUT_FILE" > "$OUTPUT_DIR/${BASENAME}.log" 2>&1; then
        if [ -f "$OUTPUT_FILE" ] && grep -q "gpu.module\|gpu.func\|gpu.thread_id\|gpu.block_id" "$OUTPUT_FILE"; then
            echo "    ✓ SUCCESS"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "    ✗ FAILED (no GPU dialect in output)"
            FAILED=$((FAILED + 1))
            FAILED_FILES+=("$BASENAME")
        fi
    else
        echo "    ✗ FAILED (conversion error)"
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("$BASENAME")
    fi
    echo ""
done

echo "=========================================="
echo "Conversion Summary"
echo "=========================================="
echo "Total tests:    $TOTAL"
echo "Successful:     $SUCCESS"
echo "Failed:         $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed tests:"
    for FAILED_FILE in "${FAILED_FILES[@]}"; do
        echo "  - $FAILED_FILE"
        echo "    Log: $OUTPUT_DIR/${FAILED_FILE}.log"
    done
    echo ""
fi

echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
