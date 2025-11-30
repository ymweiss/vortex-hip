#!/bin/bash
# Auto-generated build script by hip_splitter.py
# Compiles vecadd using split HIP compilation pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building vecadd..."


echo "Compiling kernel: vecadd_kernel..."

# Stage 1: HIP → GPU dialect MLIR
$REPO_ROOT/scripts/polygeist/hip-to-gpu-dialect.sh \
    $SCRIPT_DIR/vecadd_kernel_kernel.hip

# Stage 2: GPU dialect → Vortex MLIR
$REPO_ROOT/Polygeist/build/bin/polygeist-opt \
    --convert-gpu-to-vortex \
    $SCRIPT_DIR/vecadd_kernel_kernel.mlir \
    -o $SCRIPT_DIR/vecadd_kernel_kernel_vortex.mlir

# Stage 3: Extract metadata
python3 $REPO_ROOT/scripts/extract_kernel_metadata.py \
    $SCRIPT_DIR/vecadd_kernel_kernel_vortex.mlir \
    > $SCRIPT_DIR/vecadd_kernel_metadata.h

# Stage 4: MLIR → LLVM IR
$REPO_ROOT/Polygeist/build/bin/mlir-translate \
    --mlir-to-llvmir \
    $SCRIPT_DIR/vecadd_kernel_kernel_vortex.mlir \
    -o $SCRIPT_DIR/vecadd_kernel_kernel.ll

# Stage 5: LLVM IR → RISC-V binary
$REPO_ROOT/Polygeist/build/llvm-project/build/bin/llc \
    -march=riscv32 \
    -mattr=+m,+a,+f,+d \
    $SCRIPT_DIR/vecadd_kernel_kernel.ll \
    -o $SCRIPT_DIR/vecadd_kernel_kernel.s

# TODO: Assemble and embed kernel binary

echo "Compiling host code..."
# TODO: Implement host compilation once runtime library is ready
# clang++ vecadd_host.cpp -lhip_vortex_runtime -lvortex -o vecadd

echo "Build complete!"
echo "Kernel binaries: vecadd_kernel_kernel.s"
