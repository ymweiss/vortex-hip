# HIP-to-Vortex Compilation Pipeline

**Goal:** Enable HIP (Heterogeneous-compute Interface for Portability) applications to run on Vortex RISC-V GPU hardware through an automated compilation pipeline.

---

## Quick Start

### Prerequisites

- Linux (tested on Ubuntu 22.04, 24.04)
- GCC 11 or Clang 14+
- CMake 3.20+
- Ninja (recommended) or Make
- Python 3.8+
- storage space: 200 GB

### Clone and Initialize

```bash
git clone --recursive https://github.com/YOUR_USERNAME/vortex_hip.git
cd vortex_hip

# If already cloned without --recursive:
git submodule update --init --recursive
```

### Automated Setup (Recommended)

For a fresh clone, use the automated setup script to build all dependencies:

```bash
./scripts/setup_dependencies.sh
```

This script builds all components in the correct order:
1. Vortex (runtime + toolchain)
2. Polygeist (HIP → MLIR compiler)
3. llvm-vortex (LLVM IR → RISC-V compiler)
4. HIP runtime library

**Options:**
```bash
./scripts/setup_dependencies.sh --help           # Show all options
./scripts/setup_dependencies.sh --skip-sudo      # Skip sudo commands (if already installed)
./scripts/setup_dependencies.sh --force-vortex   # Force rebuild Vortex
./scripts/setup_dependencies.sh --force-polygeist # Force rebuild Polygeist
./scripts/setup_dependencies.sh --force-all      # Force rebuild everything
```

After setup completes, set up the environment:
```bash
source vortex/build/ci/toolchain_env.sh
```

---

## Build Instructions (Manual)

All build steps must be completed in order. Set `VORTEX_HIP_HOME` first:

```bash
export VORTEX_HIP_HOME=$PWD
```

### Step 1: Build Vortex (Runtime + Toolchain)

Polygeist converts HIP/CUDA source to MLIR. It requires LLVM, MLIR, and Clang to be built first.

See [Polygeist/README.md](Polygeist/README.md) for full build options. Quick build:

```bash
cd $VORTEX_HIP_HOME/vortex

# Install system dependencies (requires sudo)
sudo ./ci/install_dependencies.sh

# Configure build
mkdir -p build && cd build
../configure --xlen=32 --tooldir=$HOME/tools

# Install prebuilt toolchain (RISC-V compiler, libc, etc.)
./ci/toolchain_install.sh --all

# Set environment variables (REQUIRED before any compilation)
source ./ci/toolchain_env.sh

# Build Vortex runtime and kernel libraries
make -s

# Verify
ls runtime/libvortex.so kernel/libvortex.a
```

### Step 2: Build Polygeist (HIP → MLIR Compiler)

```bash
cd $VORTEX_HIP_HOME/Polygeist

# Build LLVM/MLIR/Clang (takes 30-60 minutes)
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

# Verify
./bin/cgeist --version
./bin/polygeist-opt --help | grep convert-gpu-to-vortex
```

### Step 3: Build llvm-vortex (LLVM IR → RISC-V Compiler)

```bash
cd $VORTEX_HIP_HOME/llvm-vortex

mkdir -p build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_BUILD_TYPE=Release
ninja clang llc

# Verify
./bin/llc --version | grep riscv
```

### Step 4: Build HIP Runtime Library

```bash
cd $VORTEX_HIP_HOME/runtime

mkdir -p build && cd build
cmake .. -DVORTEX_ROOT=$VORTEX_HIP_HOME/vortex -DBUILD_EXAMPLES=OFF
make

# Verify
ls libhip_vortex.so
```

---

## Compiling HIP Programs

After completing all build steps, use the automated compilation script:

```bash
# Set environment (do this each session)
export VORTEX_HIP_HOME=/path/to/vortex_hip  # Replace with your actual path
cd $VORTEX_HIP_HOME
source vortex/build/ci/toolchain_env.sh     # Sets up PATH for RISC-V tools

# Compile a HIP program
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip

# Output files:
#   hip_tests/vecadd              - Host executable
#   hip_tests/vecadd_kernel.vxbin - Device kernel binary
#   hip_tests/vecadd_kernel_args.h - Generated kernel launch stub
```

### Script Options

```bash
./scripts/compile_hip_v2.sh <input.hip> [options]

Options:
  -o <name>       Output executable name (default: input basename)
  --device-only   Only compile device code (kernel binary + stubs)
  --host-only     Only compile host code (requires existing stubs)
  --keep-temps    Keep intermediate files for debugging
  --verbose       Show all commands
```

### Running on Vortex Simulator

```bash
# Set runtime environment
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH=$VORTEX_HIP_HOME/vortex/build/runtime:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VORTEX_HIP_HOME/runtime/build:$LD_LIBRARY_PATH

# Run (kernel.vxbin must be in current directory)
./vecadd
```

---

## How It Works

The compilation pipeline transforms HIP kernel launches for Vortex:

1. **Source Transform** (`cgeist --transform-hip-source`):
   - Generates `__launch_<kernel>()` wrapper function
   - Wrapper calls generated stub on host, `<<<>>>` on device
   - Extracts kernel argument metadata from AST

2. **Device Compilation**:
   - Polygeist converts HIP → GPU MLIR → Vortex MLIR → LLVM IR
   - `--convert-gpu-to-vortex` generates `<kernel>_args.h` stub with argument metadata
   - llvm-vortex compiles LLVM IR → RISC-V `.vxbin` binary

3. **Host Compilation** (with `HIP_HOST_COMPILATION` defined):
   - Wrapper calls `launch_<kernel>()` from generated stub
   - Stub uses `vortexLaunchKernel()` with metadata for proper argument marshaling
   - Host pointers (64-bit) are converted to device addresses (32-bit)

4. **Runtime**:
   - `vortexLaunchKernel()` loads `./<kernel>.vxbin` on first launch
   - Arguments are marshaled using metadata (pointer conversion, offset mapping)
   - Kernel is cached for subsequent launches

---

## Compiling HIP Kernels for Vortex

The compilation pipeline uses **split compilation** - host and device code are compiled separately, with metadata generated during device compilation that is used by the host compilation.

### Overview

```
                           HIP Source (.hip)
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  cgeist --transform-hip-src │
                    │  (AST-level transformation) │
                    └─────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            transformed.cu   *_args.h    __launch_* wrapper
            (conditional)    (stub)      (#ifdef HOST/DEVICE)
                    │             │             │
        ┌───────────┴─────────────┴─────────────┘
        │
        │  ┌──────────────────────────────────────────────┐
        │  │              DEVICE PATH                      │
        │  │              (Polygeist)                      │
        │  └──────────────────────────────────────────────┘
        │
        ├──► cgeist (GPU MLIR) ──► polygeist-opt ──► mlir-opt
        │                              │
        │                              ▼
        │                    *_args.h (stub with metadata)
        │                    *_kernel.meta.json
        │                              │
        │         ┌────────────────────┘
        │         │
        │         ▼
        │    mlir-translate ──► llc ──► clang (link)
        │                                   │
        │                                   ▼
        │                           <kernel>.vxbin
        │
        │  ┌──────────────────────────────────────────────┐
        │  │              HOST PATH                        │
        │  │    (g++ with HIP_HOST_COMPILATION)           │
        │  └──────────────────────────────────────────────┘
        │
        └──► transformed.cu + *_args.h ──► g++ ──► executable
                         │
                         │  __launch_* wrapper calls
                         │  launch_*() from stub which
                         │  uses vortexLaunchKernel()
                         │
                         ▼
                  host_executable
```

### Step 1: Write HIP Source

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

void launch_vecadd(float* a, float* b, float* c, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vecadd_kernel<<<numBlocks, blockSize>>>(a, b, c, n);
}
```

### Step 2: Transform Source for Split Compilation

Use `inject_kernel_launchers.py` to prepare the source for both host and device compilation:

```bash
python3 scripts/polygeist/inject_kernel_launchers.py vecadd.hip vecadd_transformed.cu
```

> **Note:** The `.cu` extension is required because Polygeist's CUDA frontend expects `.cu` files.
> The content is still HIP code - this is purely a parsing technicality.

**What this script does:**
- Reorganizes source for split compilation:
  - Device header and preprocessor defines at top (gated with `#ifdef __CUDA__`)
  - Kernel definitions (gated with `#ifdef __CUDA__`)
  - Synthetic launch wrappers (gated with `#ifdef __CUDA__`)
  - All host code (includes, main, etc.) gated with `#ifndef __CUDA__`
- This ensures Polygeist only sees kernel code when compiling with `--cuda-lower`

**Generated structure:**
```cpp
// Device header and defines (Polygeist sees this via -I runtime/device)
#ifdef __CUDA__
#include "hip_runtime.h"  // -> runtime/device/hip_runtime.h (CUDA builtins)
#include <stdint.h>
#endif

// Kernel definitions (device only)
#ifdef __CUDA__
__global__ void vecadd_kernel(...) { ... }
#endif

// Host code (excluded from device compilation)
#ifndef __CUDA__
#include <hip/hip_runtime.h>  // -> runtime/host/hip/hip_runtime.h
#include <iostream>
...
int main() { ... }
#endif

// Polygeist launch wrapper (device only)
#ifdef __CUDA__
void __polygeist_launch_vecadd_kernel(...) {
    vecadd_kernel<<<blocks, threads>>>(...);
}
#endif
```

### Step 3: Device Compilation (Kernel → RISC-V Binary)

#### 3a. HIP → GPU Dialect MLIR

```bash
./Polygeist/build/bin/cgeist vecadd_transformed.cu \
    --cuda-lower \
    --emit-cuda \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -resource-dir=./Polygeist/llvm-project/build/lib/clang/18 \
    -I./runtime/device \
    -I. \
    --function='*' \
    --output-intermediate-gpu=1 \
    -S \
    -o vecadd_gpu.mlir
```

#### 3b. GPU Dialect → Vortex MLIR (+ Metadata Generation)

```bash
./Polygeist/build/bin/polygeist-opt vecadd_gpu.mlir \
    --convert-gpu-to-vortex \
    -o vecadd_vortex.mlir
```

The `--convert-gpu-to-vortex` pass uses the `kernel_arg_mapping` attribute (generated by KernelOutlining) to identify user arguments and generate correct metadata.

**This pass generates metadata files:**
- `<kernel_name>.meta.json` - JSON metadata for runtime
- `<kernel_name>_args.h` - C header with argument structure

**Example metadata (`launch_vecadd.meta.json`):**
```json
{
  "kernel_name": "launch_vecadd",
  "arguments": [
    {"name": "arg0", "type": "i32", "size": 4, "offset": 0, "is_pointer": false},
    {"name": "arg1", "type": "i32", "size": 4, "offset": 4, "is_pointer": false},
    {"name": "arg2", "type": "i32", "size": 4, "offset": 8, "is_pointer": false},
    {"name": "arg3", "type": "ptr", "size": 4, "offset": 12, "is_pointer": true},
    {"name": "arg4", "type": "ptr", "size": 4, "offset": 16, "is_pointer": true},
    {"name": "arg5", "type": "ptr", "size": 4, "offset": 20, "is_pointer": true}
  ],
  "total_args_size": 24,
  "architecture": "rv32"
}
```

#### 3c. Generate Host Stubs from Metadata

```bash
python3 scripts/generate_host_stubs.py *.meta.json -o kernel_stubs.h
```

**What this script does:**
- Reads `.meta.json` files from device compilation
- Generates C++ header with:
  - Packed argument structures (`vecadd_args_t`)
  - Metadata arrays (`vecadd_metadata[]`)
  - Type-safe launcher functions (`launch_vecadd()`)

**Generated `kernel_stubs.h`:**
```cpp
#include <vortex_hip_runtime.h>

// Argument structure matching Vortex kernel layout
typedef struct __attribute__((packed)) {
  int32_t arg0;   // grid info (inserted by runtime)
  int32_t arg1;
  int32_t arg2;
  void* arg3;     // user arg: float* a
  void* arg4;     // user arg: float* b
  void* arg5;     // user arg: float* c
} launch_vecadd_args_t;

// Metadata for runtime argument marshaling
static const VortexKernelArgMeta launch_vecadd_metadata[] = {
  { .offset = 0,  .size = 4, .is_pointer = 0 },
  { .offset = 4,  .size = 4, .is_pointer = 0 },
  { .offset = 8,  .size = 4, .is_pointer = 0 },
  { .offset = 12, .size = 4, .is_pointer = 1 },
  { .offset = 16, .size = 4, .is_pointer = 1 },
  { .offset = 20, .size = 4, .is_pointer = 1 },
};

// Type-safe launcher (call this from host code)
inline hipError_t launch_vecadd(
    dim3 gridDim, dim3 blockDim,
    int32_t arg0, int32_t arg1, int32_t arg2,
    const void* arg3, const void* arg4, const void* arg5) {
  launch_vecadd_args_t args = {arg0, arg1, arg2, (void*)arg3, (void*)arg4, (void*)arg5};
  return vortexLaunchKernel("launch_vecadd", gridDim, blockDim,
                            &args, sizeof(args),
                            launch_vecadd_metadata, 6);
}
```

#### 3d. MLIR Lowering → LLVM Dialect

```bash
./Polygeist/llvm-project/build/bin/mlir-opt vecadd_vortex.mlir \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    -o vecadd_llvm.mlir
```

#### 3e. Generate Vortex Main Wrapper

```bash
./Polygeist/build/bin/polygeist-opt vecadd_llvm.mlir \
    --generate-vortex-main \
    -o vecadd_with_main.mlir
```

#### 3f. MLIR → LLVM IR

```bash
./Polygeist/llvm-project/build/bin/mlir-translate \
    --mlir-to-llvmir \
    vecadd_with_main.mlir \
    -o vecadd.ll
```

#### 3g. LLVM IR → RISC-V Binary

The LLVM IR generated by `mlir-translate` has an x86_64 target triple by default.
We use `llc` to compile to RISC-V object code with the correct target triple,
then link with clang.

```bash
# Set environment variables for manual compilation
# These paths are set by vortex/build/ci/toolchain_install.sh during Vortex build
export VORTEX_HIP_HOME=/path/to/vortex_hip          # Root of this repository
export VORTEX_HOME=$VORTEX_HIP_HOME/vortex          # Vortex submodule
export LLVM_VORTEX=$VORTEX_HIP_HOME/llvm-vortex/build  # llvm-vortex build directory
export LIBC_VORTEX=$HOME/tools/libc32               # RISC-V libc (from toolchain_install.sh)
export LIBCRT_VORTEX=$HOME/tools/libcrt32           # RISC-V compiler-rt (from toolchain_install.sh)
export RISCV_TOOLCHAIN=$HOME/tools/riscv32-gnu-toolchain  # RISC-V GNU toolchain

# Step 1: Compile LLVM IR to RISC-V object with llc
# This sets the correct target triple (mlir-translate outputs x86_64)
$LLVM_VORTEX/bin/llc \
    --mtriple=riscv32-unknown-unknown-elf \
    -march=riscv32 \
    -mattr=+m,+a,+f \
    -filetype=obj \
    vecadd.ll \
    -o vecadd.o

# Step 2: Link with Vortex runtime
# Note: -z norelro is required to avoid .got section placement issues
$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -mcmodel=medany \
    -fno-rtti -fno-exceptions \
    -nostartfiles -nostdlib \
    vecadd.o \
    -Wl,-Bstatic,--gc-sections,-z,norelro \
    -Wl,-T,$VORTEX_HOME/kernel/scripts/link32.ld \
    -Wl,--defsym=STARTUP_ADDR=0x80000000 \
    $VORTEX_HOME/build/kernel/libvortex.a \
    -L$LIBC_VORTEX/lib -lm -lc \
    $LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a \
    -o kernel.elf

# Step 3: Convert ELF to Vortex binary format
# vxbin.py prepends VMA range header and extracts raw binary
export OBJCOPY=$RISCV_TOOLCHAIN/bin/riscv32-unknown-elf-objcopy
python3 $VORTEX_HOME/kernel/scripts/vxbin.py kernel.elf kernel.vxbin
```

**What vxbin.py does:**
- Uses `readelf` to determine the VMA (Virtual Memory Address) range
- Uses RISC-V `objcopy` to extract raw binary from ELF
- Prepends 16-byte header with min_vma and max_vma (64-bit little endian each)
- The resulting `.vxbin` file is what the Vortex runtime loads

### Step 4: Host Compilation

The host compilation uses the transformed source and generated `kernel_stubs.h`:

```bash
# Compile host code from transformed source
# -DVORTEX_HIP_HOST gates out device code, includes host HIP runtime
g++ -x c++ -c vecadd_transformed.cu -o vecadd_host.o \
    -I $VORTEX_HIP_HOME/runtime/host \
    -I .  \
    -std=c++17 \
    -DVORTEX_HIP_HOST

# Link with HIP runtime and Vortex runtime
g++ vecadd_host.o -o vecadd \
    -L $VORTEX_HIP_HOME/runtime/build \
    -L $VORTEX_HIP_HOME/vortex/build/runtime \
    -lhip_vortex -lvortex \
    -Wl,-rpath,$VORTEX_HIP_HOME/runtime/build \
    -Wl,-rpath,$VORTEX_HIP_HOME/vortex/build/runtime
```

The transformed source (`vecadd_transformed.cu`) contains both host and device code, with `#ifndef __CUDA__` / `#ifdef __CUDA__` guards. The `-DVORTEX_HIP_HOST` define ensures only host code is compiled by g++.

---

## Split Compilation Model

### Why Split Compilation?

- **Simpler toolchain** - standard g++ for host, no custom frontend needed
- **Better separation** - host and device are independent compilation units
- **Metadata bridge** - device compilation generates metadata used by host
- **Industry standard** - CUDA/HIP/OpenCL all use this model

### Key Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `compile_hip_v2.sh` | **Full pipeline automation** | `.hip` | executable + `.vxbin` |
| `cgeist --transform-hip-source` | AST-level source transformation | `.hip` | `_transformed.cu` + `*_args.h` |

### Quick Compilation (Automated)

Use the `compile_hip_v2.sh` script to run the entire pipeline automatically:

```bash
# Basic usage
./scripts/compile_hip_v2.sh my_kernel.hip

# With options
./scripts/compile_hip_v2.sh my_kernel.hip -o my_app --verbose

# Device compilation only (generates stubs and kernel binary)
./scripts/compile_hip_v2.sh my_kernel.hip --device-only

# Keep intermediate files for debugging
./scripts/compile_hip_v2.sh my_kernel.hip --keep-temps
```

**Options:**
- `-o <name>` - Output executable name (default: input basename)
- `--device-only` - Only compile device code
- `--host-only` - Only compile host code
- `--keep-temps` - Keep intermediate files
- `--verbose` - Show all commands being executed

### Metadata Flow

```
Device Compilation                    Host Compilation
─────────────────                    ────────────────
     │                                     │
     ▼                                     │
--convert-gpu-to-vortex                    │
     │                                     │
     ├──► .meta.json ──► generate_host_stubs.py ──► kernel_stubs.h
     │                                     │                │
     ▼                                     │                │
kernel.vxbin                               ▼                ▼
     │                              g++ vecadd_host.cpp ◄───┘
     │                                     │
     └─────────────────────────────────────┤
                                           ▼
                                    host_executable
```

### LLVM 18 Throughout

The pipeline uses LLVM 18 for all components:

```
Polygeist (LLVM 18)              llvm-vortex (LLVM 18)
        ↓                                ↓
  HIP → GPU MLIR                   LLVM IR → RISC-V
        ↓                                ↑
        └────── LLVM IR (.ll) ───────────┘
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
│   ├── host/                     # Host-side HIP runtime headers
│   │   ├── hip_vortex_runtime.h  # Main host runtime (HIP API, vortexLaunchKernel)
│   │   └── hip/
│   │       └── hip_runtime.h     # Compatibility wrapper
│   ├── device/                   # Device-side headers for Polygeist
│   │   └── hip_runtime.h         # CUDA builtins (__global__, threadIdx, etc.)
│   └── src/
│       └── vortex_hip_runtime.cpp  # Runtime implementation
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

### ✅ Phase 2: Kernel Compilation (Complete)

**Completed:**
- ✅ Polygeist compiles HIP kernels to GPU dialect MLIR
- ✅ 23/23 test kernels compile successfully (100% pass rate)
- ✅ **End-to-end vecadd test passes on Vortex SimX**
- ✅ HIP source transformation with conditional wrapper generation
- ✅ Split compilation model with proper argument marshaling
- ✅ **`--convert-gpu-to-vortex` pass** - Lowers GPU intrinsics to Vortex:
  - `gpu.thread_id` → `vx_get_threadIdx()` TLS accessor
  - `gpu.block_id` → `vx_get_blockIdx()` TLS accessor
  - `gpu.block_dim` → `vx_get_blockDim()` TLS accessor
  - `gpu.grid_dim` → `vx_get_gridDim()` TLS accessor
  - `gpu.barrier` → `vx_barrier(bar_id, num_threads)`
  - `printf` → `vx_printf`
  - Kernel extraction from `gpu.module` to `func.func`
  - Kernel metadata extraction (JSON + C header generation)
- ✅ **Kernel argument ordering fix**:
  - `ReorderGPUKernelArgsPass` matches kernels by arg count
  - `vortex.kernel_name` attribute propagates original name
  - Host stub uses `vortexLaunchKernel()` with metadata
  - 64-bit host pointers converted to 32-bit device addresses
- ✅ **`--generate-vortex-main` pass** - Generates Vortex entry point:
  - `main()` function that reads args from `VX_CSR_MSCRATCH` (0x340)
  - `kernel_body(void* args)` wrapper that unpacks arguments
  - `vx_spawn_threads()` integration for thread dispatch

**GPU Dialect Operations Lowered:**
- `gpu.block_id`, `gpu.thread_id` - thread indexing → Vortex TLS
- `gpu.block_dim`, `gpu.grid_dim` - dimension queries → Vortex TLS
- `gpu.barrier` - synchronization → `vx_barrier_abi()`
- `gpu.launch_func` - kernel launches → metadata extraction
- `gpu.module`, `gpu.func` - kernel extraction → `func.func`
- `memref.global` (address space 3) - shared memory globals → offset annotations
- `memref.get_global` (address space 3) - shared memory access → `VX_CSR_LOCAL_MEM_BASE` + offset

**Memory Model:**
- ✅ Global memory (address space 0) - standard memref lowering via `--finalize-memref-to-llvm`
- ✅ Shared memory (address space 3) - custom CSR-based lowering via `VX_CSR_LOCAL_MEM_BASE` (0xFC3)

### 🔄 Phase 3: Full Integration (In Progress)

- ✅ End-to-end compilation pipeline (`compile_hip_v2.sh`)
- ⏳ Testing remaining kernels on SimX (sgemm, conv3, etc.)
- ⏳ Performance validation with complex kernels
- ⏳ Extended HIP API coverage

---

## Running on Vortex Simulator

Once you have compiled both host and kernel binaries, you can run them on the Vortex simulator.

### Prerequisites

Before running, ensure you have:
1. Built the Vortex runtime (see [Build Vortex Runtime Library](#build-vortex-runtime-library))
2. Set up the environment: `source vortex/build/ci/toolchain_env.sh`
3. Compiled host binary and kernel binary (`.vxbin`)

### Running with SimX Simulator

SimX is Vortex's software simulator (fastest, good for debugging):

```bash
# Set library paths and driver
export VORTEX_RT_PATH=$PWD/vortex/build/runtime
export LD_LIBRARY_PATH=$VORTEX_RT_PATH:$LD_LIBRARY_PATH
export VORTEX_DRIVER=simx

# Run the application (kernel.vxbin must be in current directory or specify path)
./my_hip_app

# Or specify kernel path explicitly in your host code
```

### Running with RTL Simulator

RTL simulation runs the actual hardware design (slower, cycle-accurate):

```bash
export VORTEX_RT_PATH=$PWD/vortex/build/runtime
export LD_LIBRARY_PATH=$VORTEX_RT_PATH:$LD_LIBRARY_PATH
export VORTEX_DRIVER=rtlsim

./my_hip_app
```

### Complete Example: vecadd

The easiest way to compile is using the automated script:

```bash
# 1. Setup environment
cd vortex_hip
source vortex/build/ci/toolchain_env.sh

# 2. Compile using automated script (recommended)
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip --verbose

# This generates:
#   hip_tests/vecadd              - Host executable
#   hip_tests/vecadd_kernel.vxbin - Device kernel binary
#   hip_tests/vecadd_kernel_args.h - Kernel launch stub

# 3. Run on simulator
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH=$PWD/vortex/build/runtime:$PWD/runtime/build:$LD_LIBRARY_PATH
./hip_tests/vecadd

# Expected output:
# [HIP] vortexLaunchKernel: Loading kernel: ./vecadd_kernel.vxbin
# ...
# PASSED!
```

#### Manual Compilation Steps (for reference)

If you need to run the pipeline manually:

```bash
# Step 1: Transform source (gates host code, extracts kernels)
python3 scripts/polygeist/inject_kernel_launchers.py \
    hip_tests/vecadd.hip \
    /tmp/vecadd_transformed.cu

# Step 2: HIP → GPU MLIR
./Polygeist/build/bin/cgeist /tmp/vecadd_transformed.cu \
    --cuda-lower \
    --emit-cuda \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -resource-dir=./Polygeist/llvm-project/build/lib/clang/18 \
    -I./runtime/device \
    -I. \
    --function='*' \
    --output-intermediate-gpu=1 \
    -S \
    -o /tmp/vecadd_gpu.mlir

# Step 3: GPU MLIR → Vortex MLIR (generates metadata files)
./Polygeist/build/bin/polygeist-opt /tmp/vecadd_gpu.mlir \
    --convert-gpu-to-vortex \
    -o /tmp/vecadd_vortex.mlir

# Step 4: Generate host stubs from metadata
python3 scripts/generate_host_stubs.py *.meta.json -o kernel_stubs.h

# Step 5: MLIR lowering → LLVM Dialect
./Polygeist/llvm-project/build/bin/mlir-opt /tmp/vecadd_vortex.mlir \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    -o /tmp/vecadd_llvm.mlir

# Step 6: Generate Vortex main wrapper
./Polygeist/build/bin/polygeist-opt /tmp/vecadd_llvm.mlir \
    --generate-vortex-main \
    -o /tmp/vecadd_with_main.mlir

# Step 7: MLIR → LLVM IR
./Polygeist/llvm-project/build/bin/mlir-translate \
    --mlir-to-llvmir \
    /tmp/vecadd_with_main.mlir \
    -o /tmp/vecadd.ll

# Step 8: LLVM IR → RISC-V object (fixes target triple)
$LLVM_VORTEX/bin/llc \
    --mtriple=riscv32-unknown-unknown-elf \
    -march=riscv32 \
    -mattr=+m,+a,+f \
    -filetype=obj \
    /tmp/vecadd.ll \
    -o /tmp/vecadd.o

# Step 9: Link to ELF
$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -mcmodel=medany \
    -fno-rtti -fno-exceptions \
    -nostartfiles -nostdlib \
    /tmp/vecadd.o \
    -Wl,-Bstatic,--gc-sections,-z,norelro \
    -Wl,-T,$VORTEX_HOME/kernel/scripts/link32.ld \
    -Wl,--defsym=STARTUP_ADDR=0x80000000 \
    $VORTEX_HOME/build/kernel/libvortex.a \
    -L$LIBC_VORTEX/lib -lm -lc \
    $LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a \
    -o kernel.elf

# Step 10: ELF → Vortex binary format
python3 $VORTEX_HOME/kernel/scripts/vxbin.py kernel.elf kernel.vxbin
```

### Kernel Binary Discovery

The Vortex runtime looks for kernel binaries in these locations:
1. The path specified when calling `vortexRegisterKernel()`
2. The current working directory (for `kernel.vxbin`)
3. Embedded in the host executable (if using binary embedding)

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VORTEX_DRIVER` | Backend driver | `simx`, `rtlsim`, `opae`, `xrt` |
| `VORTEX_RT_PATH` | Runtime library path | `/path/to/vortex/build/runtime` |
| `LD_LIBRARY_PATH` | Library search path | Must include `$VORTEX_RT_PATH` |

### Debugging Tips

1. **Check kernel loading:**
   ```bash
   VORTEX_DEBUG=1 ./my_hip_app  # Enable debug output
   ```

2. **Verify kernel binary:**
   ```bash
   file kernel.vxbin  # Should show RISC-V ELF
   $LLVM_VORTEX/bin/llvm-objdump -d kernel.elf  # Disassemble
   ```

3. **Check runtime initialization:**
   - Ensure `VORTEX_DRIVER` is set correctly
   - Ensure runtime libraries are in `LD_LIBRARY_PATH`

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

Compile all kernels using the automated script:

```bash
for kernel in hip_tests/*.hip; do
    ./scripts/compile_hip.sh "$kernel" --device-only
done
```

Or compile manually to MLIR:

```bash
for kernel in hip_tests/*.hip; do
    name=$(basename "$kernel" .hip)
    python3 scripts/polygeist/inject_kernel_launchers.py "$kernel" "/tmp/${name}.cu"
    ./Polygeist/build/bin/cgeist "/tmp/${name}.cu" \
        --cuda-lower --emit-cuda --cuda-gpu-arch=sm_60 \
        -nocudalib -nocudainc \
        -resource-dir=./Polygeist/llvm-project/build/lib/clang/18 \
        -I./runtime/device -I. \
        --function='*' --output-intermediate-gpu=1 -S \
        -o "hip_tests/mlir_output/${name}_kernel.mlir"
done
```

---

## Key Technical Details

### Why Split Compilation?

- **Simpler toolchain** - standard g++ for host, no custom frontend
- **Better separation** - host and device are independent
- **Easier debugging** - test each path independently
- **Industry standard** - CUDA/HIP/OpenCL all use this model

### LLVM 18 Pipeline

```
Polygeist (LLVM 18)              llvm-vortex (LLVM 18)
        ↓                                ↓
  HIP → GPU MLIR                   LLVM IR → RISC-V
        ↓                                ↑
        └────── LLVM IR (.ll) ───────────┘
```

### HIP Runtime Header Structure

The runtime headers are organized into host and device directories:

```
runtime/
├── host/                          # Host-side headers (for g++ compilation)
│   ├── hip_vortex_runtime.h       # Main runtime: HIP API + vortexLaunchKernel
│   └── hip/
│       └── hip_runtime.h          # Compatibility wrapper (includes hip_vortex_runtime.h)
│
└── device/                        # Device-side headers (for Polygeist)
    └── hip_runtime.h              # CUDA builtins: __global__, threadIdx, blockIdx, etc.
```

**Host compilation** (g++ with `-I runtime/host`):
- `#include <hip/hip_runtime.h>` → `runtime/host/hip/hip_runtime.h`
- Provides HIP API (hipMalloc, hipMemcpy, etc.) mapped to Vortex runtime

**Device compilation** (Polygeist with `-I runtime/device`):
- Uses `runtime/device/hip_runtime.h`
- Provides CUDA builtins that Polygeist understands (`__global__`, `threadIdx`, etc.)

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
| llvm-vortex | RISC-V code generation | LLVM 18 |

---

**Last Updated:** 2025-12-19
**Current Phase:** Phase 3 (Full Integration - In Progress)
