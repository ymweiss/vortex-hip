# HIP-to-Vortex Compilation Pipeline

**Goal:** Enable HIP (Heterogeneous-compute Interface for Portability) applications to run on Vortex RISC-V GPU hardware through an automated compilation pipeline.

---

## Quick Start

### Prerequisites

- Linux (tested on Ubuntu 22.04, 24.04)
- GCC 11+ or Clang 14+
- CMake 3.20+
- Ninja (recommended) or Make
- Python 3.8+
- Storage space: ~200 GB

### Clone and Initialize

```bash
git clone --recursive https://github.com/YOUR_USERNAME/vortex_hip.git
cd vortex_hip

# If already cloned without --recursive:
git submodule update --init --recursive
```

### Automated Setup (Recommended)

```bash
./scripts/setup_dependencies.sh
```

This builds all components in order:
1. Vortex (runtime + toolchain)
2. Polygeist (HIP → MLIR compiler)
3. llvm-vortex (LLVM IR → RISC-V compiler)
4. HIP runtime library

After setup:
```bash
source vortex/build/ci/toolchain_env.sh
```

---

## Compiling HIP Programs

### Basic Usage

```bash
# Set environment (do this each session)
source vortex/build/ci/toolchain_env.sh

# Compile a HIP program
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip -o vecadd

# Output files in hip_tests/build_vecadd/:
#   vecadd                  - Host executable
#   vecadd_kernel.vxbin     - Device kernel binary
#   vecadd_kernel_args.h    - Generated kernel launch stub
```

### Compilation Options

```bash
./scripts/compile_hip_v2.sh <input.hip> [options]

Options:
  -o <name>       Output executable name (default: input basename)
  --device-only   Only compile device code (kernel binary + stubs)
  --host-only     Only compile host code (requires existing stubs)
  --keep-temps    Keep intermediate files for debugging
  --verbose       Show all commands

Environment Variables:
  XLEN=32|64      Target pointer width (default: 32)
                  Set XLEN=64 for 64-bit Vortex (RV64)
```

### 64-bit Vortex Support

```bash
# Compile for RV64 Vortex
XLEN=64 ./scripts/compile_hip_v2.sh hip_tests/vecadd.hip -o vecadd64
```

### Running on Vortex Simulator

```bash
# Set runtime environment
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH=$PWD/vortex/build/runtime:$PWD/runtime/build:$LD_LIBRARY_PATH

# Run (from the build directory containing the .vxbin file)
cd hip_tests/build_vecadd
./vecadd
```

---

## How It Works

The compilation pipeline transforms HIP source files for Vortex in a single automated flow:

```
                        HIP Source (.hip/.cu)
                                │
                                ▼
               ┌────────────────────────────────┐
               │   cgeist --transform-hip-source │
               │   (AST-level transformation)    │
               └────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        _transformed.cu    *_args.h     __launch_* wrapper
                │          (stub)
                ▼
               ┌────────────────────────────────┐
               │       cgeist --cuda-lower       │
               │       (HIP → GPU MLIR)          │
               └────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────┐
               │ polygeist-opt                   │
               │   --convert-gpu-to-vortex       │
               │   (GPU MLIR → Vortex MLIR)      │
               └────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────┐
               │ mlir-opt (LLVM lowering)        │
               │ polygeist-opt --generate-       │
               │   vortex-main                   │
               │ mlir-translate --mlir-to-llvmir │
               └────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────┐
               │ llc + clang (llvm-vortex)       │
               │ (LLVM IR → RISC-V binary)       │
               └────────────────────────────────┘
                                │
                                ▼
                        kernel.vxbin
```

### Pipeline Stages

1. **Source Transformation** (`cgeist --transform-hip-source`):
   - Parses HIP source and extracts kernel signatures from AST
   - Generates `__launch_<kernel>()` wrapper functions
   - Generates `*_args.h` stub headers with correct argument metadata
   - Replaces `hipLaunchKernelGGL()` calls with wrapper calls

2. **Device Compilation** (Polygeist):
   - `cgeist --cuda-lower`: HIP → GPU dialect MLIR
   - `--convert-gpu-to-vortex`: GPU intrinsics → Vortex CSR reads
   - `--generate-vortex-main`: Generate kernel entry point
   - llvm-vortex: LLVM IR → RISC-V `.vxbin` binary

3. **Host Compilation** (g++):
   - Compiles transformed source with `HIP_HOST_COMPILATION` defined
   - Wrapper calls generated stub using `vortexLaunchKernel()`
   - Links with HIP runtime library

4. **Runtime**:
   - `vortexLaunchKernel()` loads `./<kernel>.vxbin` on first launch
   - Arguments marshaled using metadata (pointer conversion, offset mapping)
   - Kernel cached for subsequent launches

---

## Writing HIP Code

Use standard HIP syntax with `hipLaunchKernelGGL`:

```cpp
#include <hip/hip_runtime.h>

__global__ void vecadd_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // ... allocate memory with hipMalloc ...

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(vecadd_kernel,
                       dim3(numBlocks), dim3(blockSize),
                       0, 0,
                       d_a, d_b, d_c, n);

    // ... copy results with hipMemcpy ...
}
```

### Supported Features

- `threadIdx.x/y/z`, `blockIdx.x/y/z`, `blockDim.x/y/z`, `gridDim.x/y/z`
- `__syncthreads()` barriers
- `__shared__` memory
- `printf()` from device code
- 1D, 2D, and 3D kernel dispatch
- Multiple kernels per source file

---

## Test Status

All 23 test kernels pass (100%):

| Kernel | Features |
|--------|----------|
| vecadd | Basic 1D thread indexing |
| sgemm | 2D matrix multiply |
| sgemm2 | Tiled matmul with shared memory + barriers |
| dotproduct | Parallel reduction with shared memory |
| cta | 2D grid dispatch |
| stencil3d | 3D kernel dispatch |
| printf | Device-side printf |
| diverge | Control flow divergence |
| conv3 | 2D convolution |
| sort | Multi-kernel bitonic sort |
| dogfood | Multi-kernel stress test |
| ... | See hip_tests/ for all tests |

Run all tests:
```bash
for kernel in hip_tests/*.hip; do
    ./scripts/compile_hip_v2.sh "$kernel" --device-only
done
```

---

## Project Structure

```
vortex_hip/
├── Polygeist/                    # HIP → MLIR compiler (submodule)
│   └── build/bin/
│       ├── cgeist               # C++/CUDA/HIP → MLIR frontend
│       └── polygeist-opt        # MLIR optimizer with Vortex passes
│
├── llvm-vortex/                  # RISC-V backend with Vortex support
│   └── build/bin/
│       ├── llc                  # LLVM IR → RISC-V compiler
│       └── clang                # Linker
│
├── vortex/                       # Vortex GPU (submodule)
│   └── build/
│       ├── runtime/             # Host runtime (libvortex.so)
│       └── kernel/              # Device runtime (libvortex.a)
│
├── runtime/
│   ├── host/                    # Host-side HIP headers
│   │   ├── hip_vortex_runtime.h # HIP API + vortexLaunchKernel
│   │   └── hip/hip_runtime.h    # Compatibility wrapper
│   ├── device/                  # Device-side headers for Polygeist
│   │   └── hip_runtime.h        # CUDA builtins (__global__, threadIdx, etc.)
│   └── src/
│       └── vortex_hip_runtime.cpp
│
├── scripts/
│   └── compile_hip_v2.sh        # Main compilation script
│
└── hip_tests/                   # Test kernels
```

---

## Build Instructions (Manual)

### Step 1: Build Vortex

```bash
cd vortex
sudo ./ci/install_dependencies.sh
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools
./ci/toolchain_install.sh --all
source ./ci/toolchain_env.sh
make -s
```

### Step 2: Build Polygeist

```bash
cd Polygeist

# Build LLVM/MLIR/Clang
mkdir -p llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release
ninja
cd ../..

# Build Polygeist
mkdir -p build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release
ninja cgeist polygeist-opt
```

### Step 3: Build llvm-vortex

```bash
cd llvm-vortex
mkdir -p build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_BUILD_TYPE=Release
ninja clang llc
```

### Step 4: Build HIP Runtime

```bash
cd runtime
mkdir -p build && cd build
cmake .. -DVORTEX_ROOT=$PWD/../../vortex
make
```

---

## GPU Dialect Lowering

The `--convert-gpu-to-vortex` pass lowers GPU intrinsics:

| GPU Operation | Vortex Lowering |
|---------------|-----------------|
| `gpu.thread_id` | CSR read + TLS offset calculation |
| `gpu.block_id` | CSR read from spawn args |
| `gpu.block_dim` | CSR read from spawn args |
| `gpu.grid_dim` | CSR read from spawn args |
| `gpu.barrier` | `vx_barrier()` call |
| `__syncthreads()` | `vx_barrier()` call |
| Shared memory | `VX_CSR_LOCAL_MEM_BASE` + offset |

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `XLEN` | Target pointer width | `32`, `64` |
| `VORTEX_DRIVER` | Backend driver | `simx`, `rtlsim`, `opae`, `xrt` |
| `LD_LIBRARY_PATH` | Runtime library path | Must include vortex/build/runtime |

---

## Debugging

### Keep Intermediate Files
```bash
./scripts/compile_hip_v2.sh input.hip --keep-temps --verbose
```

### Check Kernel Binary
```bash
file kernel.vxbin
$LLVM_VORTEX/bin/llvm-objdump -d kernel.elf
```

### Enable Runtime Debug
```bash
VORTEX_DEBUG=1 ./my_app
```

---

## External Dependencies

| Submodule | Purpose | Version |
|-----------|---------|---------|
| Polygeist | HIP → MLIR compiler | LLVM 18 |
| vortex | Target GPU hardware | master |
| llvm-vortex | RISC-V code generation | LLVM 18 |

---

**Last Updated:** 2025-12-27
