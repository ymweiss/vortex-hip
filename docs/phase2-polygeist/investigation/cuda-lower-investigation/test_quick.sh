#!/bin/bash
cd /home/yaakov/vortex_hip/Polygeist/cuda-lower-investigation

RESDIR=/home/yaakov/vortex_hip/Polygeist/llvm-project/build/lib/clang/18/

/home/yaakov/vortex_hip/Polygeist/build/bin/cgeist \
  /home/yaakov/vortex_hip/Polygeist/tools/cgeist/Test/CUDA/polybench-cuda/gemver/gemver.cu \
  --function=kernel_A -S --cuda-lower \
  --nocudainc --nocudalib \
  -resource-dir=$RESDIR \
  -o kernel_test.mlir 2>&1 | head -100

echo "---"
echo "Exit code: $?"
ls -lh kernel_test.mlir 2>&1 || echo "File not created"
