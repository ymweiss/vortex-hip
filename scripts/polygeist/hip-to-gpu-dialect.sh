#!/bin/bash
# Script to convert HIP/CUDA code to GPU dialect IR using Polygeist
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input.cu/.hip> [output.mlir]"
    echo "Example: $0 hip_tests/simple_kernel.hip gpu_dialect.mlir"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.cu}.mlir}"
OUTPUT_FILE="${OUTPUT_FILE%.hip}.mlir"

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POLYGEIST_SRC="$REPO_ROOT/Polygeist"
CGEIST="$POLYGEIST_SRC/build/bin/cgeist"
RESOURCE_DIR="$POLYGEIST_SRC/llvm-project/build/lib/clang/18"
HIP_HEADERS="$REPO_ROOT/runtime/include"
CLANG_CUDA_BUILTINS="$POLYGEIST_SRC/tools/cgeist/Test/Verification/Inputs"

if [ ! -f "$CGEIST" ]; then
    echo "Error: cgeist not found at $CGEIST"
    echo "Please build Polygeist first using scripts/polygeist/build-polygeist.sh"
    exit 1
fi

echo "Converting $INPUT_FILE to GPU dialect IR..."
echo "Output: $OUTPUT_FILE"

$CGEIST "$INPUT_FILE" \
    --cuda-gpu-arch=sm_60 \
    -nocudalib \
    -nocudainc \
    -resource-dir="$RESOURCE_DIR" \
    -I"$CUDA_HEADERS" \
    --function=* \
    --emit-cuda \
    -S \
    -o "$OUTPUT_FILE"

echo "GPU dialect IR generated successfully!"
echo ""
echo "To view the output:"
echo "  cat $OUTPUT_FILE"
