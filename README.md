# HIP-to-Vortex Compilation Pipeline

**Goal:** Enable HIP (Heterogeneous-compute Interface for Portability) applications to run on Vortex RISC-V GPU hardware through an automated compilation pipeline.

**Status:** Phase 1 (Runtime) complete, Phase 2 (Kernel Compilation) in progress

---

## Quick Start

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- GCC 11+ or Clang 14+
- CMake 3.20+
- Ninja (recommended) or Make
- Python 3.8+

### Clone and Initialize

```bash
git clone --recursive https://github.com/YOUR_USERNAME/vortex_hip.git
cd vortex_hip

# If already cloned without --recursive:
git submodule update --init --recursive
```

### Build Polygeist (HIP to MLIR Compiler)

Polygeist converts HIP/CUDA source to MLIR. It requires LLVM, MLIR, and Clang to be built first.

See [Polygeist/README.md](Polygeist/README.md) for full build options. Quick build:

```bash
cd Polygeist

# 1. Build LLVM/MLIR/Clang dependencies
mkdir -p llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release
ninja    # Takes 30-60 minutes
cd ../..

# 2. Build Polygeist
mkdir -p build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release
ninja cgeist polygeist-opt
cd ..

# Verify
./build/bin/cgeist --version
```

### Build Vortex Runtime Library

The host executable links against `libvortex.so` to communicate with the Vortex simulator/hardware.

```bash
cd vortex

# Install system dependencies (requires sudo)
sudo ./ci/install_dependencies.sh

# Configure build
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools

# Install prebuilt toolchain (RISC-V compiler, LLVM, etc.)
./ci/toolchain_install.sh --all

# Set environment variables (run this before each session)
source ./ci/toolchain_env.sh

# Build runtime libraries
make -C runtime

# Verify libraries were built
ls -la runtime/libvortex*.so
```

**Output libraries:**
- `libvortex.so` - Core runtime API
- `libvortex-simx.so` - SimX simulator backend
- `libvortex-rtlsim.so` - RTL simulator backend

**Linking host code:**
```bash
g++ host_code.o -o app \
    -L/path/to/vortex/build/runtime \
    -L/path/to/vortex_hip/runtime/hip_vortex_runtime/lib \
    -lvortex -lvortex-simx -lhip_vortex_runtime \
    -Wl,-rpath,/path/to/vortex/build/runtime
```

### Build llvm-vortex (Device Code Compiler)

llvm-vortex is the LLVM backend that compiles LLVM IR to Vortex RISC-V binaries. This is needed to convert device code from MLIR/LLVM IR to executable kernel binaries.

```bash
cd llvm-vortex

mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --target clang llc

# Verify
./bin/llc --version | grep riscv
```

**Usage (after GPUToVortex lowering produces LLVM IR):**
```bash
# Compile LLVM IR to Vortex RISC-V object
./llvm-vortex/build/bin/llc -march=riscv32 -mcpu=generic-rv32 \
    -mattr=+m,+f kernel.ll -o kernel.o

# Link to create kernel binary
riscv32-unknown-elf-ld kernel.o -o kernel.vxbin
```

---

## Compiling HIP Kernels for Vortex

### Step 1: Prepare HIP Source

Write your HIP kernel using standard HIP syntax:

```cpp
// vecadd.hip
#include <hip/hip_runtime.h>

__global__ void vecadd_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Step 2: Transform Source for Split Compilation

Use the `inject_kernel_launchers.py` script to prepare the source:

```bash
python3 scripts/polygeist/inject_kernel_launchers.py vecadd.hip vecadd_transformed.cu
```

This script:
- Adds conditional includes (`#ifndef __CUDA__` for host, `#else` for device)
- Wraps kernel definitions for dual compilation
- Injects synthetic launch wrappers required by Polygeist

### Step 3: Compile Kernel to GPU Dialect MLIR

```bash
./Polygeist/build/bin/cgeist vecadd_transformed.cu \
    --cuda-lower \
    --cuda-gpu-arch=sm_60 \
    -nocudalib \
    -nocudainc \
    -resource-dir=./Polygeist/llvm-project/build/lib/clang/18 \
    -I. \
    --function='*' \
    --emit-cuda \
    -S \
    -o vecadd.mlir
```

### Step 4: Compile Host Code

```bash
g++ -c vecadd_transformed.cu -o vecadd_host.o \
    -I runtime/hip_vortex_runtime/include \
    -std=c++17
```

---

## Split Compilation Model

The pipeline uses split compilation - host and device code are compiled separately:

```
                    HIP Source (.hip)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │ HOST PATH     │           │ DEVICE PATH   │
    │ (g++)         │           │ (Polygeist)   │
    ├───────────────┤           ├───────────────┤
    │ hip/hip_      │           │ hip_runtime_  │
    │ runtime.h     │           │ vortex/       │
    │ (Vortex API)  │           │ hip_runtime.h │
    ├───────────────┤           ├───────────────┤
    │ extern kernel │           │ full kernel   │
    │ declarations  │           │ definitions   │
    └───────┬───────┘           └───────┬───────┘
            │                           │
            ▼                           ▼
    host_executable             kernel.mlir (GPU dialect)
            │                           │
            │                           ▼
            │                   [--convert-gpu-to-vortex]
            │                   (Vortex intrinsics + metadata)
            │                           │
            │                           ▼
            │                   [--gpu-to-llvm]
            │                           │
            │                           ▼
            │                   [--generate-vortex-main]
            │                   (main() + kernel_body wrapper)
            │                           │
            │                           ▼
            │                   [mlir-translate --mlir-to-llvmir]
            │                           │
            │                           ▼
            │                   [llvm-vortex clang → RISC-V]
            │                           │
            │                           ▼
            │                   kernel.vxbin
            │                           │
            └───────────┬───────────────┘
                        │
                        ▼
                   Runtime loads
                   kernel binary
```

### Device Compilation Pipeline

```bash
# 1. HIP → GPU dialect MLIR
./Polygeist/build/bin/cgeist kernel.hip --emit-cuda -S -o kernel.mlir

# 2. GPU dialect → Vortex LLVM dialect
./Polygeist/build/bin/polygeist-opt kernel.mlir \
    --convert-gpu-to-vortex \
    --gpu-to-llvm \
    --generate-vortex-main \
    -o kernel_vortex.mlir

# 3. MLIR LLVM Dialect → LLVM IR (textual)
# mlir-translate is built with Polygeist's LLVM
./Polygeist/llvm-project/build/bin/mlir-translate \
    --mlir-to-llvmir kernel_vortex.mlir -o kernel.ll

# 4. LLVM IR → Vortex RISC-V binary
llvm-vortex/build/bin/clang -target riscv32 kernel.ll -o kernel.vxbin
```

---

## Project Structure

```
vortex_hip/
├── Polygeist/                    # HIP → MLIR compiler (submodule)
│   └── build/bin/
│       ├── cgeist               # C++/CUDA → MLIR frontend
│       └── polygeist-opt        # MLIR optimizer
│
├── vortex/                       # Vortex GPU (submodule)
│
├── scripts/
│   └── polygeist/
│       └── inject_kernel_launchers.py  # Source transformation
│
├── runtime/
│   └── hip_vortex_runtime/       # HIP API → Vortex runtime mapping
│       ├── include/hip/
│       │   └── hip_runtime.h     # HIP API declarations
│       └── src/
│           └── hip_runtime.cpp   # Runtime implementation
│
├── hip_runtime_vortex/
│   └── hip_runtime.h            # Device-side header for Polygeist
│
├── hip_tests/
│   ├── kernels/                 # HIP kernel sources
│   └── gpu_mlir_output/         # Generated MLIR files
│
└── docs/
    ├── PHASES_OVERVIEW.md       # Project phases
    ├── GPU_TO_VORTEX_LOWERING.md # Lowering reference
    └── WORK_DISTRIBUTION.md     # Implementation tasks
```

---

## Current Status

### ✅ Phase 1: HIP Runtime (Complete)

- HIP runtime library mapping to Vortex API
- 13 runtime tests passing on Vortex SimX simulator
- Memory management, device management, kernel launch

### 🔄 Phase 2: Kernel Compilation (90% Complete)

**Completed:**
- ✅ Polygeist compiles HIP kernels to GPU dialect MLIR
- ✅ 21/22 test kernels compile successfully
- ✅ Source transformation script (`inject_kernel_launchers.py`)
- ✅ Split compilation model validated
- ✅ **`--convert-gpu-to-vortex` pass** - Lowers GPU intrinsics to Vortex:
  - `gpu.thread_id` → `vx_get_threadIdx()` TLS accessor
  - `gpu.block_id` → `vx_get_blockIdx()` TLS accessor
  - `gpu.block_dim` → `vx_get_blockDim()` TLS accessor
  - `gpu.grid_dim` → `vx_get_gridDim()` TLS accessor
  - `gpu.barrier` → `vx_barrier(bar_id, num_threads)`
  - `printf` → `vx_printf`
  - Kernel metadata extraction (JSON + C header generation)
- ✅ **`--generate-vortex-main` pass** - Generates Vortex entry point:
  - `main()` function that reads args from `VX_CSR_MSCRATCH` (0x340)
  - `kernel_body(void* args)` wrapper that unpacks arguments
  - `vx_spawn_threads()` integration for thread dispatch

**GPU Dialect Operations Lowered:**
- `gpu.module`, `gpu.func` - kernel definitions
- `gpu.block_id`, `gpu.thread_id` - thread indexing → Vortex TLS
- `gpu.block_dim`, `gpu.grid_dim` - dimension queries → Vortex TLS
- `gpu.barrier` - synchronization → `vx_barrier()`
- `gpu.launch_func` - kernel launches → metadata extraction

**Remaining:**
- End-to-end integration testing
- Vortex SimX simulator testing
- Performance optimization

### ⏳ Phase 3: Full Integration (Planned)

- End-to-end compilation pipeline automation
- Performance optimizations
- Extended HIP API coverage

---

## Test Kernels

The `hip_tests/kernels/` directory contains test kernels:

| Kernel | Features |
|--------|----------|
| vecadd | Basic thread indexing |
| sgemm | Matrix multiplication |
| sgemm2 | Shared memory, barriers |
| printf | Device-side printf |
| diverge | Control flow divergence |
| conv3 | 3D convolution |

Compile all kernels to MLIR:

```bash
for kernel in hip_tests/kernels/*.hip; do
    name=$(basename "$kernel" .hip)
    python3 scripts/polygeist/inject_kernel_launchers.py "$kernel" "/tmp/${name}.cu"
    ./Polygeist/build/bin/cgeist "/tmp/${name}.cu" \
        --cuda-lower --cuda-gpu-arch=sm_60 \
        -nocudalib -nocudainc \
        -resource-dir=./Polygeist/llvm-project/build/lib/clang/18 \
        -I. --function='*' --emit-cuda -S \
        -o "hip_tests/gpu_mlir_output/${name}.mlir"
done
```

---

## Key Technical Details

### Why Split Compilation?

- **Simpler toolchain** - standard g++ for host, no custom frontend
- **Better separation** - host and device are independent
- **Easier debugging** - test each path independently
- **Industry standard** - CUDA/HIP/OpenCL all use this model

### No LLVM Version Conflicts

```
Polygeist (LLVM 18)              llvm-vortex (LLVM 10)
        ↓                                ↓
  HIP → GPU MLIR                   LLVM IR → RISC-V
        ↓                                ↑
        └────── LLVM IR (.ll) ───────────┘
                (version-independent)
```

### Macro-Based Header Selection

The `inject_kernel_launchers.py` script uses `__CUDA__` macro (defined by Polygeist's clang) to select headers:

```cpp
#ifndef __CUDA__
#include "hip/hip_runtime.h"           // Host: Vortex runtime API
#else
#include "hip_runtime_vortex/hip_runtime.h"  // Device: CUDA builtins
#endif
```

---

## Documentation

- **[docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)** - Complete phase breakdown
- **[docs/GPU_TO_VORTEX_LOWERING.md](docs/GPU_TO_VORTEX_LOWERING.md)** - GPU dialect lowering reference
- **[docs/WORK_DISTRIBUTION.md](docs/WORK_DISTRIBUTION.md)** - Implementation tasks

---

## External Dependencies

| Submodule | Purpose | Version |
|-----------|---------|---------|
| Polygeist | HIP → MLIR compiler | LLVM 18 |
| vortex | Target GPU hardware | master |
| llvm-vortex | RISC-V code generation | LLVM 10 |

---

**Last Updated:** 2025-12-05
**Current Phase:** Phase 2 (Kernel Compilation - 90% Complete)
