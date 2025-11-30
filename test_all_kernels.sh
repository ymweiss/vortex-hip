#!/bin/bash
failed=0
passed=0
for f in hip_tests/mlir_output/*_kernel.mlir; do
  if [[ "$f" =~ vortex ]]; then
    continue
  fi
  base=$(basename "$f" .mlir)
  echo "Testing $base..."
  if ./Polygeist/build/bin/polygeist-opt --convert-gpu-to-vortex "$f" -o "hip_tests/mlir_output/${base}_vortex_new.mlir" 2>&1; then
    ((passed++))
  else
    echo "FAILED: $base"
    ((failed++))
  fi
done
echo "Results: $passed passed, $failed failed out of $((passed+failed)) kernels"
