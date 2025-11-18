#!/bin/bash
# Batch convert all HIP kernel files to GPU dialect MLIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KERNEL_DIR="$REPO_ROOT/hip_tests/kernels"
OUTPUT_DIR="$REPO_ROOT/hip_tests/mlir_output"
CONVERTER="$SCRIPT_DIR/hip-to-gpu-dialect.sh"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "HIP Kernel to GPU Dialect Batch Converter"
echo "=========================================="
echo ""
echo "Kernel directory: $KERNEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Find all .hip files in kernels directory
KERNEL_FILES=$(find "$KERNEL_DIR" -name "*.hip" 2>/dev/null | sort)

if [ -z "$KERNEL_FILES" ]; then
    echo "No .hip files found in $KERNEL_DIR"
    exit 1
fi

TOTAL=0
SUCCESS=0
FAILED=0

for KERNEL_FILE in $KERNEL_FILES; do
    TOTAL=$((TOTAL + 1))
    BASENAME=$(basename "$KERNEL_FILE" .hip)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.mlir"

    echo "[$TOTAL] Converting: $(basename "$KERNEL_FILE")"

    if "$CONVERTER" "$KERNEL_FILE" "$OUTPUT_FILE" > /tmp/convert_$$.log 2>&1; then
        if grep -q "GPU dialect operations found" /tmp/convert_$$.log; then
            SUCCESS=$((SUCCESS + 1))
            echo "    ✓ Success: $OUTPUT_FILE"
        else
            FAILED=$((FAILED + 1))
            echo "    ⚠ Warning: No GPU dialect operations found"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "    ✗ Failed"
        echo "    Error log:"
        tail -5 /tmp/convert_$$.log | sed 's/^/      /'
    fi
    echo ""
done

rm -f /tmp/convert_$$.log

echo "=========================================="
echo "Conversion Summary"
echo "=========================================="
echo "Total files:     $TOTAL"
echo "Successful:      $SUCCESS"
echo "Failed:          $FAILED"
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All conversions successful!"
    exit 0
else
    echo "⚠ Some conversions failed"
    exit 1
fi
