#!/bin/bash
set -e

# Limit parallel jobs to 8 threads
export NINJA_STATUS="[%f/%t %p %es] "
THREADS=8

# Get the repository root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POLYGEIST_SRC="$REPO_ROOT/Polygeist"
LLVM_SRC="$POLYGEIST_SRC/llvm-project"
BUILD_DIR="$POLYGEIST_SRC/llvm-project/build"

# Check if LLVM is already built
if [ -f "$BUILD_DIR/bin/clang" ] && [ -f "$BUILD_DIR/bin/mlir-opt" ]; then
  echo "LLVM+MLIR+Clang already built, skipping..."
else
  # Create build directory
  mkdir -p $BUILD_DIR

  # Configure LLVM with MLIR, using lld linker and minimal features for faster build
  echo "Configuring LLVM with MLIR..."
  cd $BUILD_DIR
  cmake -G Ninja $LLVM_SRC/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INSTALL_UTILS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

  # Build LLVM+MLIR+Clang (limited to 8 threads)
  echo "Building LLVM, MLIR, and Clang (using $THREADS threads)..."
  ninja -j$THREADS
fi

# Configure Polygeist with CUDA syntax-only support
# This enables CUDA/HIP kernel syntax parsing without requiring CUDA toolkit
echo "Configuring Polygeist..."
cd $POLYGEIST_SRC
mkdir -p build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir \
  -DLLVM_DIR=$BUILD_DIR/lib/cmake/llvm \
  -DCLANG_DIR=$BUILD_DIR/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DPOLYGEIST_USE_LINKER=lld \
  -DPOLYGEIST_ENABLE_CUDA_SYNTAX_ONLY=ON \
  -DPOLYGEIST_ENABLE_ROCM=0 \
  -DPOLYGEIST_ENABLE_POLYMER=0 \
  -DBUILD_SHARED_LIBS=OFF

# Build Polygeist (limited to 8 threads)
echo "Building Polygeist (using $THREADS threads)..."
ninja -j$THREADS

echo "Build complete!"
echo "cgeist binary: $POLYGEIST_SRC/build/bin/cgeist"
