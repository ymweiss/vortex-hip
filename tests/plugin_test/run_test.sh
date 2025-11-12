#!/bin/bash
# Test script for HIPMetadataExtractor plugin

set -e

LLVM_BUILD="/home/yaakov/vortex_hip/llvm-vortex/build"
PLUGIN_SO="${LLVM_BUILD}/lib/HIPMetadataExtractor.so"
CLANG="${LLVM_BUILD}/bin/clang++"
TEST_DIR="/home/yaakov/vortex_hip/tests/plugin_test"

echo "========================================================================"
echo "HIPMetadataExtractor Plugin Test"
echo "========================================================================"
echo ""

# Check if plugin exists
if [ ! -f "${PLUGIN_SO}" ]; then
    echo "❌ Plugin not found: ${PLUGIN_SO}"
    echo "Please build the plugin first:"
    echo "  cd ${LLVM_BUILD}"
    echo "  ninja HIPMetadataExtractor"
    exit 1
fi

echo "✅ Plugin found: ${PLUGIN_SO}"
echo ""

# Check if clang++ exists
if [ ! -f "${CLANG}" ]; then
    echo "❌ Clang++ not found: ${CLANG}"
    echo "Please build clang first:"
    echo "  cd ${LLVM_BUILD}"
    echo "  ninja clang"
    exit 1
fi

echo "✅ Clang++ found: ${CLANG}"
echo ""

# Run the plugin on the test kernel
echo "Running plugin on test_kernel.hip..."
echo "------------------------------------------------------------------------"

"${CLANG}" \
  -x hip \
  -Xclang -load \
  -Xclang "${PLUGIN_SO}" \
  -Xclang -plugin \
  -Xclang hip-metadata \
  -Xclang -plugin-arg-hip-metadata \
  -Xclang -output \
  -Xclang -plugin-arg-hip-metadata \
  -Xclang "${TEST_DIR}/test_metadata.cpp" \
  -fsyntax-only \
  --cuda-device-only \
  --no-cuda-version-check \
  "${TEST_DIR}/test_kernel.hip" 2>&1 | grep -v "error: no such file" | grep -v ".amdgcn.bc" || true

echo ""
echo "------------------------------------------------------------------------"
echo ""

# Verify the generated metadata
if [ ! -f "${TEST_DIR}/test_metadata.cpp" ]; then
    echo "❌ Metadata file not generated"
    exit 1
fi

echo "✅ Metadata file generated: test_metadata.cpp"
echo ""
echo "Verifying metadata..."
echo "------------------------------------------------------------------------"

python3 "${TEST_DIR}/verify_metadata.py" "${TEST_DIR}/test_metadata.cpp"
