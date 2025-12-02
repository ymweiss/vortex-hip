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

```bash
cd Polygeist

# Configure (uses Ninja by default)
cmake -G Ninja -B build \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release

# Build (takes 30-60 minutes)
cmake --build build --target cgeist polygeist-opt

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
            │                   [GPUToVortex pass]
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

### 🔄 Phase 2: Kernel Compilation (In Progress)

**Completed:**
- ✅ Polygeist compiles HIP kernels to GPU dialect MLIR
- ✅ 21/22 test kernels compile successfully
- ✅ Source transformation script (`inject_kernel_launchers.py`)
- ✅ Split compilation model validated

**GPU Dialect Operations Generated:**
- `gpu.module`, `gpu.func` - kernel definitions
- `gpu.block_id`, `gpu.thread_id` - thread indexing
- `gpu.barrier` - thread synchronization (when shared memory used)
- `gpu.launch_func` - kernel launches

**Remaining:**
- GPUToVortex MLIR pass (lower GPU dialect to Vortex)
- LLVM IR generation
- Vortex binary generation

### ⏳ Phase 3: Full Integration (Planned)

- End-to-end compilation pipeline
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

**Last Updated:** 2025-11-30
**Current Phase:** Phase 2 (Kernel Compilation)
