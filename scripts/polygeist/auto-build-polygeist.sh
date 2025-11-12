#!/bin/bash
set -e

# Automated build script that proceeds from LLVM to Polygeist
THREADS=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POLYGEIST_SRC="$REPO_ROOT/Polygeist"
BUILD_DIR=$POLYGEIST_SRC/llvm-project/build

echo "=== Automated Polygeist Build ===" | tee -a auto-build.log
echo "Started: $(date)" | tee -a auto-build.log

# Wait for LLVM build to complete
echo "" | tee -a auto-build.log
echo "Waiting for LLVM/MLIR/Clang build to complete..." | tee -a auto-build.log

while true; do
    if [ -f "$BUILD_DIR/bin/clang" ] && [ -f "$BUILD_DIR/bin/mlir-opt" ]; then
        echo "✓ LLVM dependencies built!" | tee -a auto-build.log
        break
    fi

    # Check if ninja is still running
    if ! pgrep -f "ninja -j8" > /dev/null; then
        echo "⚠️  Ninja stopped but dependencies not complete. Checking for errors..." | tee -a auto-build.log
        if [ ! -f "$BUILD_DIR/bin/clang" ]; then
            echo "ERROR: clang not built. Check build.log for errors." | tee -a auto-build.log
            exit 1
        fi
        if [ ! -f "$BUILD_DIR/bin/mlir-opt" ]; then
            echo "ERROR: mlir-opt not built. Check build.log for errors." | tee -a auto-build.log
            exit 1
        fi
    fi

    sleep 60  # Check every minute
done

echo "" | tee -a auto-build.log
echo "LLVM build completed: $(date)" | tee -a auto-build.log

# Configure Polygeist
echo "" | tee -a auto-build.log
echo "Configuring Polygeist..." | tee -a auto-build.log
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
  -DPOLYGEIST_ENABLE_CUDA=0 \
  -DPOLYGEIST_ENABLE_ROCM=0 \
  -DPOLYGEIST_ENABLE_POLYMER=0 \
  -DBUILD_SHARED_LIBS=OFF | tee -a ../auto-build.log

echo "" | tee -a ../auto-build.log
echo "Building Polygeist (using $THREADS threads)..." | tee -a ../auto-build.log
ninja -j$THREADS | tee -a ../auto-build.log

echo "" | tee -a ../auto-build.log
echo "=== Build Complete ===" | tee -a ../auto-build.log
echo "Finished: $(date)" | tee -a ../auto-build.log

if [ -f "$POLYGEIST_SRC/build/bin/cgeist" ]; then
    echo "✓✓✓ SUCCESS: cgeist built successfully!" | tee -a ../auto-build.log
    echo "Binary location: $POLYGEIST_SRC/build/bin/cgeist" | tee -a ../auto-build.log
else
    echo "ERROR: cgeist binary not found!" | tee -a ../auto-build.log
    exit 1
fi
