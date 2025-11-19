# Vortex Compilation Flow and Architecture Analysis

## Document Purpose

This document describes how Vortex compiles GPU programs (both host and kernel code) and integrates with the compiler toolchain. It covers two compilation approaches:

1. **Standard Vortex Compilation**: Traditional approach using the vx_spawn framework and custom Vortex API
2. **HIP Integration**: New approach using HIP/CUDA syntax for portability (in development, 70% complete)

Understanding both flows is critical for:
- Working with existing Vortex applications
- Implementing the HIP→Vortex compilation pipeline
- Ensuring compatibility between the two approaches

---

## Table of Contents

- [1. Overview: Two Compilation Approaches](#1-overview-two-compilation-approaches)
- [2. Standard Vortex Compilation](#2-standard-vortex-compilation)
  - [2.1 Host Code Compilation (x86)](#21-host-code-compilation-x86)
  - [2.2 Kernel Code Compilation (RISC-V)](#22-kernel-code-compilation-risc-v)
  - [2.3 Host-Kernel Execution Flow](#23-host-kernel-execution-flow)
- [3. HIP-Based Compilation](#3-hip-based-compilation)
  - [3.1 Host Code Compilation](#31-host-code-compilation)
  - [3.2 Kernel Code Compilation](#32-kernel-code-compilation)
  - [3.3 Host-Kernel Separation in MLIR](#33-host-kernel-separation-in-mlir)
- [4. Compiler Toolchain Integration](#4-compiler-toolchain-integration)
- [5. Binary Format and Loading](#5-binary-format-and-loading)
- [6. Thread/Warp/Core Model](#6-threadwarpcore-model)
- [7. Comparison: Standard vs HIP](#7-comparison-standard-vs-hip)
- [8. Key Integration Requirements for HIP](#8-key-integration-requirements-for-hip)

---

## 1. OVERVIEW: TWO COMPILATION APPROACHES

Vortex supports two distinct programming models:

### 1.1 Standard Vortex (vx_spawn framework)

**Status**: Production-ready, fully implemented

**Characteristics**:
- Separate host and kernel source files
- Explicit host API (`vortex.h`) for device control
- Kernel API (`vx_spawn.h`) for thread management
- Direct LLVM compilation (no MLIR)
- Manual memory management

**Use Case**: Native Vortex applications, maximum control

### 1.2 HIP Integration

**Status**: In development (70% kernel-side, 30% host-side remaining)

**Characteristics**:
- Unified source files (`.hip`) with host and device code
- HIP/CUDA-compatible API (portable to NVIDIA/AMD GPUs)
- MLIR-based compilation pipeline via Polygeist
- Automatic lowering to Vortex primitives
- GPU dialect intermediate representation

**Use Case**: Portable GPU applications, compatibility with HIP/CUDA ecosystem

### 1.3 Compilation Model Comparison

Both approaches produce the same final output:
- **Host binary** (x86 or RISC-V host architecture)
- **Kernel binary** (.vxbin format, RISC-V with +vortex extensions)
- Runtime linkage via **libvortex.so**

---

## 2. STANDARD VORTEX COMPILATION

### 2.1 Host Code Compilation (x86)

#### 2.1.1 Host Program Structure

**Example**: `vortex/tests/regression/vecadd/main.cpp`

```cpp
#include <vortex.h>  // Vortex runtime API
#include <vector>

int main() {
    const uint32_t count = 1024;

    // 1. Device Initialization
    vx_device_h device;
    vx_dev_open(&device);

    // 2. Allocate device memory
    vx_buffer_h src0_buffer, src1_buffer, dst_buffer;
    size_t buf_size = count * sizeof(int32_t);

    vx_mem_alloc(device, buf_size, VX_MEM_READ, &src0_buffer);
    vx_mem_alloc(device, buf_size, VX_MEM_READ, &src1_buffer);
    vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer);

    // Get device addresses
    uint64_t src0_addr, src1_addr, dst_addr;
    vx_mem_address(src0_buffer, &src0_addr);
    vx_mem_address(src1_buffer, &src1_addr);
    vx_mem_address(dst_buffer, &dst_addr);

    // 3. Prepare host data
    std::vector<int32_t> h_src0(count), h_src1(count), h_dst(count);
    for (uint32_t i = 0; i < count; ++i) {
        h_src0[i] = i;
        h_src1[i] = i * 2;
    }

    // 4. Upload data to device
    vx_copy_to_dev(src0_buffer, h_src0.data(), 0, buf_size);
    vx_copy_to_dev(src1_buffer, h_src1.data(), 0, buf_size);

    // 5. Prepare kernel arguments
    kernel_arg_t kernel_arg;
    kernel_arg.src0_addr = src0_addr;
    kernel_arg.src1_addr = src1_addr;
    kernel_arg.dst_addr = dst_addr;
    kernel_arg.num_points = count;

    // 6. Upload kernel binary
    vx_buffer_h krnl_buffer;
    vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buffer);

    // 7. Upload kernel arguments
    vx_buffer_h args_buffer;
    vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer);

    // 8. Start device execution
    vx_start(device, krnl_buffer, args_buffer);

    // 9. Wait for completion
    vx_ready_wait(device, VX_MAX_TIMEOUT);

    // 10. Download results
    vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size);

    // 11. Cleanup
    vx_mem_free(src0_buffer);
    vx_mem_free(src1_buffer);
    vx_mem_free(dst_buffer);
    vx_buf_release(krnl_buffer);
    vx_buf_release(args_buffer);
    vx_dev_close(device);

    return 0;
}
```

#### 2.1.2 Host Compilation Pipeline

```
main.cpp (C++ with vortex.h API)
    ↓
[g++ or clang++]
    - Standard C++17 compilation
    - Includes: -I$(VORTEX_HOME)/runtime/include
    - Optimization: -O3
    ↓
main.o (Object file)
    ↓
[Linker]
    - Links: -L$(VORTEX_RT_PATH) -lvortex
    - Driver-specific: -lvortex-simx / -lvortex-rtlsim / -lvortex-fpga
    ↓
executable (Host binary - x86 ELF)
    - Dynamically linked to libvortex.so
    - Controls Vortex device via runtime API
```

#### 2.1.3 Host Compilation Flags

```makefile
# Standard C++ compiler
CXX = g++  # or clang++

# Compilation flags
CXXFLAGS += -std=c++17 -Wall -Wextra -Wno-maybe-uninitialized
CXXFLAGS += -O3
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include

# Linking
LDFLAGS += -L$(VORTEX_RT_PATH) -lvortex
LDFLAGS += -pthread  # Runtime uses threads
```

#### 2.1.4 Vortex Runtime API

**Header**: `vortex/runtime/include/vortex.h`

**Device Management**:
```c
int vx_dev_open(vx_device_h* hdevice);
int vx_dev_close(vx_device_h hdevice);
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value);
```

**Memory Management**:
```c
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);
int vx_mem_free(vx_buffer_h hbuffer);
int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address);
int vx_mem_reserve(vx_device_h hdevice, uint64_t address, uint64_t size, int flags, vx_buffer_h* hbuffer);
```

**Data Transfer**:
```c
int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size);
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size);
```

**Execution Control**:
```c
int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments);
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);
```

**Utility Functions**:
```c
int vx_upload_kernel_file(vx_device_h hdevice, const char* filename, vx_buffer_h* hbuffer);
int vx_upload_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer);
int vx_buf_release(vx_buffer_h hbuffer);
```

### 2.2 Kernel Code Compilation (RISC-V)

#### 2.2.1 Kernel Program Structure

**Example**: `vortex/tests/regression/vecadd/kernel.cpp`

```cpp
#include <vx_spawn.h>
#include "common.h"

// Kernel arguments structure
typedef struct {
    uint64_t src0_addr;
    uint64_t src1_addr;
    uint64_t dst_addr;
    uint32_t num_points;
} kernel_arg_t;

// Kernel function - executed by each thread
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
    auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
    auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);

    // blockIdx.x provided by vx_spawn framework
    // Each thread processes one element
    uint32_t idx = blockIdx.x;
    if (idx < arg->num_points) {
        dst_ptr[idx] = src0_ptr[idx] + src1_ptr[idx];
    }
}

// Entry point for kernel
int main() {
    // Read kernel arguments from CSR (Control/Status Register)
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    // Spawn threads to execute kernel_body
    // Dimension 1: 1D grid
    // Grid size: arg->num_points blocks
    // Block size: 1 thread per block
    return vx_spawn_threads(1, &arg->num_points, nullptr,
                           (vx_kernel_func_cb)kernel_body, arg);
}
```

#### 2.2.2 Kernel Compilation Pipeline

```
kernel.cpp (C++ with vx_spawn.h)
    ↓
[llvm-vortex clang++]
    - Target: riscv32-unknown-elf (or riscv64)
    - March: rv32imaf (or rv64imafd)
    - Custom: -Xclang -target-feature -Xclang +vortex
    - Includes: -I$(VORTEX_HOME)/kernel/include
    - Optimization: -O3 -mcmodel=medany
    - Freestanding: -nostartfiles -nostdlib
    ↓
kernel.o (RISC-V object file)
    ↓
[llvm-vortex clang++ linker]
    - Linker script: -T link32.ld
    - Entry: --defsym=STARTUP_ADDR=0x80000000
    - Libraries: libvortex.a, libc.a, libm.a
    ↓
kernel.elf (RISC-V ELF binary)
    ↓
[llvm-objdump] (optional, for debugging)
    - Generate kernel.dump disassembly
    ↓
[vxbin.py]
    - Extract LOAD segments
    - Create header with VMA range
    ↓
kernel.vxbin (Vortex binary format)
    - Ready to upload to device
```

#### 2.2.3 Kernel Compilation Flags

```makefile
# LLVM-Vortex toolchain
VX_CC  = $(LLVM_VORTEX)/bin/clang
VX_CXX = $(LLVM_VORTEX)/bin/clang++
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump
VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

# Architecture flags
CFLAGS += -march=rv32imaf -mabi=ilp32f  # 32-bit
# CFLAGS += -march=rv64imafd -mabi=lp64d  # 64-bit

# Vortex-specific LLVM flags
LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex  # CRITICAL
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond

# Optimization and code generation
VX_CFLAGS += -O3 -mcmodel=medany
VX_CFLAGS += -fno-rtti -fno-exceptions
VX_CFLAGS += -fdata-sections -ffunction-sections
VX_CFLAGS += -nostartfiles -nostdlib
VX_CFLAGS += -mllvm -disable-loop-idiom-all

# Include paths
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include

# Linking flags
VX_LDFLAGS += -Wl,-Bstatic,--gc-sections
VX_LDFLAGS += -T,$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld
VX_LDFLAGS += --defsym=STARTUP_ADDR=$(STARTUP_ADDR)
VX_LDFLAGS += $(VORTEX_KN_PATH)/libvortex.a
```

#### 2.2.4 Build Example

```makefile
# Compile kernel
kernel.elf: kernel.cpp
    $(VX_CXX) $(VX_CFLAGS) $^ $(VX_LDFLAGS) -o $@

# Generate disassembly
kernel.dump: kernel.elf
    $(VX_DP) -D $< > $@

# Convert to vxbin
kernel.vxbin: kernel.elf
    OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@
```

### 2.3 Host-Kernel Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST PROCESS (x86)                       │
│  Executable linked to libvortex.so                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
        vx_dev_open(&device)
                       │  Opens device driver (simx/rtlsim/fpga)
                       ↓
        vx_upload_kernel_file("kernel.vxbin", &kernel_buf)
                       │  Loads RISC-V binary into device memory
                       ↓
        vx_upload_bytes(&args, sizeof(args), &args_buf)
                       │  Uploads kernel arguments
                       ↓
        vx_start(device, kernel_buf, args_buf)
                       │  Writes args address to VX_CSR_MSCRATCH
                       │  Jumps to kernel entry point (_start)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            KERNEL EXECUTION (RISC-V on Vortex)              │
│                                                             │
│  1. _start (from vx_start.S)                               │
│      - Initialize TLS (Thread-Local Storage)                │
│      - Set up stack                                         │
│      - Jump to main()                                       │
│                                                             │
│  2. main()                                                  │
│      - Read args from VX_CSR_MSCRATCH                      │
│      - Call vx_spawn_threads(grid, block, kernel_body, args)│
│                                                             │
│  3. vx_spawn_threads() (from vx_spawn.c)                   │
│      - Loop through grid dimensions                         │
│      - For each block:                                      │
│          - Calculate blockIdx.x/y/z                        │
│          - Set TLS variables                               │
│          - Spawn warps if block has multiple threads        │
│          - Call kernel_body() for each thread              │
│                                                             │
│  4. kernel_body()                                          │
│      - Access blockIdx, threadIdx from TLS                 │
│      - Perform computation                                  │
│      - Write results to memory                             │
│                                                             │
│  5. Return to main()                                       │
│      - Cleanup, return to host                             │
└─────────────────────────────────────────────────────────────┘
                       │
                       ↓
        vx_ready_wait(device, timeout)
                       │  Host polls device for completion
                       ↓
        vx_copy_from_dev(host_ptr, dst_buffer, ...)
                       │  Download results from device
                       ↓
        vx_mem_free(buffers...), vx_dev_close(device)
                       │  Cleanup
                       ↓
                   Host exits
```

---

## 3. HIP-BASED COMPILATION

### 3.1 Host Code Compilation

#### 3.1.1 HIP Host Program Example

**Example**: `hip_tests/basic.hip`

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(int* a, int* b, int* c, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_basic(int* d_src, int* d_dst, uint32_t count, uint32_t threads_per_block) {
    dim3 grid((count + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);

    hipLaunchKernelGGL(vectorAdd, grid, block, 0, 0,
                       d_src, d_dst, count);
}

int main() {
    const int count = 1024;

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    // Host allocation
    h_a = new int[count];
    h_b = new int[count];
    h_c = new int[count];

    // Device allocation
    hipMalloc(&d_a, count * sizeof(int));
    hipMalloc(&d_b, count * sizeof(int));
    hipMalloc(&d_c, count * sizeof(int));

    // Initialize and upload
    for (int i = 0; i < count; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    hipMemcpy(d_a, h_a, count * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, count * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel
    launch_basic(d_a, d_c, count, 32);

    // Synchronize and download
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, count * sizeof(int), hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}
```

#### 3.1.2 HIP Host Compilation Pipeline

**Current Status**: 30% implemented (design complete, lowering pass TODO)

```
basic.hip (Unified HIP source)
    ↓
[Polygeist --cuda-lower]
    - Parses HIP host functions as standard C++
    - Converts hipLaunchKernelGGL → gpu.launch_func
    - Preserves hipMalloc, hipMemcpy as function calls
    ↓
MLIR: Host functions (func.func)
    - func.func @launch_basic(...)
    - func.func @main()
    - Contains: arith, scf, memref operations
    - Contains: gpu.launch_func operations
    ↓
[ConvertGPUToVortex pass - HOST SIDE] ⚠️ NOT YET IMPLEMENTED
    - Extract metadata from gpu.launch_func (DONE ✓)
    - Lower gpu.launch_func to LLVM dialect:
        ↓
      LLVM: Argument marshaling
        - llvm.alloca for argument struct
        - llvm.store for each argument
        ↓
      LLVM: Kernel loading (if not pre-loaded)
        - llvm.call @vx_upload_kernel_bytes(device, binary, size)
        ↓
      LLVM: Argument upload
        - llvm.call @vx_upload_bytes(device, args_struct, size)
        ↓
      LLVM: Kernel launch
        - llvm.call @vx_start(device, kernel_handle, args_handle)
        ↓
      LLVM: Synchronization
        - llvm.call @vx_ready_wait(device, timeout)
    ↓
[Standard MLIR lowering passes]
    - --convert-func-to-llvm
    - --convert-arith-to-llvm
    - --convert-memref-to-llvm
    ↓
LLVM Dialect (pure)
    - All operations in LLVM dialect
    - Calls to vx_* runtime functions
    ↓
[mlir-translate --mlir-to-llvmir]
    - Convert MLIR LLVM dialect → LLVM IR (.ll)
    ↓
LLVM IR (.ll file)
    ↓
[Host compiler: clang++ or g++]
    - Standard x86 compilation
    - Links to libvortex.so
    ↓
Host executable (x86 ELF)
    - Dynamically linked to libvortex.so
    - Contains lowered HIP API → vx_* calls
```

**Missing Implementation (30%)**:
The `LaunchFuncOpLowering` pattern in `ConvertGPUToVortex.cpp` needs to generate the LLVM calls shown above. Currently only metadata extraction is implemented.

### 3.2 Kernel Code Compilation

#### 3.2.1 HIP Kernel Example

```cpp
__global__ void vectorAdd(int* a, int* b, int* c, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        c[idx] = a[idx] + b[idx];
    }
}
```

#### 3.2.2 HIP Kernel Compilation Pipeline

**Current Status**: 70% implemented

```
basic.hip (Unified HIP source)
    ↓
[Polygeist --cuda-lower --emit-cuda]
    - Detects __global__ functions
    - Converts to gpu.module / gpu.func
    - Lowers <<<>>> syntax to gpu.launch_func
    ↓
MLIR: GPU Dialect (kernel code)
    gpu.module @__polygeist_gpu_module {
      gpu.func @vectorAdd(...) kernel {
        %tid_x = gpu.thread_id x       // threadIdx.x
        %bid_x = gpu.block_id x        // blockIdx.x
        %bdim_x = gpu.block_dim x      // blockDim.x

        // Kernel computation with arith, scf, memref ops

        gpu.barrier                    // __syncthreads()
        gpu.return
      }
    }
    ↓
[ConvertGPUToVortex pass - KERNEL SIDE] ✓ 70% COMPLETE
    - Preprocessing: Consolidate polygeist.alternatives (DONE ✓)
    - Remove duplicate kernels (DONE ✓)
    - Lower gpu.thread_id → vx_get_threadIdx() TLS access (DONE ✓)
    - Lower gpu.block_id → vx_get_blockIdx() TLS access (DONE ✓)
    - Lower gpu.block_dim → vx_get_blockDim() TLS access (DONE ✓)
    - Lower gpu.grid_dim → vx_get_gridDim() TLS access (DONE ✓)
    - Lower gpu.barrier → llvm.call @vx_barrier (DONE ✓)
    - Extract metadata from gpu.launch_func (DONE ✓)
    ↓
MLIR: GPU func with LLVM intrinsics
    gpu.module @__polygeist_gpu_module {
      llvm.func @vx_get_threadIdx() -> !llvm.ptr
      llvm.func @vx_get_blockIdx() -> !llvm.ptr
      llvm.func @vx_barrier(i32, i32)

      gpu.func @vectorAdd(...) kernel {
        // Access threadIdx via TLS
        %tidx_ptr_call = llvm.call @vx_get_threadIdx() : () -> !llvm.ptr
        %tidx_field_ptr = llvm.getelementptr %tidx_ptr_call[0, 0]
        %tid_x_i32 = llvm.load %tidx_field_ptr : !llvm.ptr -> i32

        // Similar for blockIdx, blockDim, gridDim

        // Barrier
        llvm.call @vx_barrier(%bar_id, %num_warps) : (i32, i32) -> ()
      }
    }
    ↓
[Standard GPU lowering passes] ⚠️ NEED VERIFICATION
    - --convert-gpu-to-llvm (or custom variant)
    - gpu.func → llvm.func
    - gpu.module → plain module
    ↓
LLVM Dialect (pure RISC-V kernel)
    - llvm.func @vectorAdd(i64, i32, i32, !llvm.ptr, !llvm.ptr)
    - All operations in LLVM dialect
    - Calls to vx_get_threadIdx, vx_barrier intrinsics
    ↓
[mlir-translate --mlir-to-llvmir]
    - Convert to LLVM IR
    ↓
LLVM IR (.ll file, RISC-V target)
    ↓
[llvm-vortex clang++]
    - Target: riscv32-unknown-elf
    - March: rv32imaf +vortex
    - Links: libvortex.a (provides vx_get_* functions)
    ↓
kernel.elf (RISC-V ELF)
    ↓
[vxbin.py]
    ↓
kernel.vxbin (Vortex binary format)
```

### 3.3 Host-Kernel Separation in MLIR

Polygeist automatically separates host and kernel code in the generated MLIR:

```mlir
module attributes {
  gpu.container_module,
  llvm.target_triple = "x86_64-unknown-linux-gnu",  // Host target
  polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda"  // Will be changed to riscv
} {
  //=================================================================
  // KERNEL CODE (Device side)
  //=================================================================
  gpu.module @__polygeist_gpu_module {
    gpu.func @vectorAdd(%arg0: index, %arg1: i32, %arg2: i32,
                        %arg3: memref<?xi32>, %arg4: memref<?xi32>) kernel {
      %tid_x = gpu.thread_id x
      %bid_x = gpu.block_id x
      %bdim_x = gpu.block_dim x

      %idx = arith.addi %tid_x, ... : index
      // ... kernel computation ...

      gpu.return
    }
  }

  //=================================================================
  // HOST CODE (Host side)
  //=================================================================
  func.func @launch_basic(%arg0: memref<?xi32>, %arg1: memref<?xi32>,
                          %arg2: i32, %arg3: i32) {
    %grid_x = arith.constant 32 : index
    %block_x = arith.constant 32 : index
    %c1 = arith.constant 1 : index

    // Grid/block dimension calculations (arith ops)
    %num_blocks = arith.divsi ...

    // Launch kernel
    gpu.launch_func @__polygeist_gpu_module::@vectorAdd
      blocks in (%num_blocks, %c1, %c1)
      threads in (%block_x, %c1, %c1)
      args(%arg2 : i32, %arg3 : i32, %arg0 : memref<?xi32>, %arg1 : memref<?xi32>)
      {vortex.kernel_metadata = "..."}  // Metadata added by our pass

    return
  }

  func.func @main() {
    // Host program logic (memory allocation, data init, calls to launch_basic)
    return
  }
}
```

**Key Observations**:
1. **gpu.module** contains device code compiled for RISC-V
2. **func.func** operations contain host code compiled for x86
3. **gpu.launch_func** is the boundary between host and device
4. Two separate compilation units with different targets

---

## 7. COMPARISON: STANDARD VS HIP

### 7.1 Feature Comparison

| Feature | Standard Vortex | HIP Integration |
|---------|-----------------|-----------------|
| **Source Structure** | Separate host/kernel files | Single .hip file |
| **Host API** | `vx_*` functions (vortex.h) | `hip*` functions (hip_runtime.h) |
| **Kernel API** | `vx_spawn.h` framework | `__global__`, threadIdx, blockIdx |
| **Memory API** | `vx_mem_alloc`, `vx_copy_to_dev` | `hipMalloc`, `hipMemcpy` |
| **Launch API** | `vx_start(kernel, args)` | `hipLaunchKernelGGL(<<<grid, block>>>)` |
| **Compilation** | Direct LLVM (clang++) | Polygeist → MLIR → LLVM |
| **IR** | None (direct to LLVM IR) | GPU dialect → LLVM dialect |
| **Binary Format** | .vxbin | .vxbin (same) |
| **Runtime** | libvortex.so | libvortex.so (same) |
| **Portability** | Vortex-specific | HIP/CUDA-compatible |
| **Status** | Production ✅ | In development (70%) |

### 7.2 Code Comparison

#### Memory Allocation

**Standard Vortex**:
```cpp
vx_buffer_h buffer;
vx_mem_alloc(device, size, VX_MEM_READ, &buffer);
uint64_t dev_addr;
vx_mem_address(buffer, &dev_addr);
vx_copy_to_dev(buffer, host_ptr, 0, size);
```

**HIP**:
```cpp
int* d_ptr;
hipMalloc(&d_ptr, size);
hipMemcpy(d_ptr, host_ptr, size, hipMemcpyHostToDevice);
```

#### Kernel Launch

**Standard Vortex**:
```cpp
// Prepare argument struct
kernel_arg_t args = {.src = src_addr, .dst = dst_addr, .count = count};

// Upload kernel and arguments
vx_upload_kernel_file(device, "kernel.vxbin", &kernel_buf);
vx_upload_bytes(device, &args, sizeof(args), &args_buf);

// Launch
vx_start(device, kernel_buf, args_buf);
vx_ready_wait(device, VX_MAX_TIMEOUT);
```

**HIP**:
```cpp
dim3 grid(num_blocks);
dim3 block(threads_per_block);

hipLaunchKernelGGL(myKernel, grid, block, 0, 0,
                   d_src, d_dst, count);
hipDeviceSynchronize();
```

#### Kernel Function

**Standard Vortex**:
```cpp
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t idx = blockIdx.x;  // From vx_spawn TLS
    if (idx < arg->count) {
        arg->dst[idx] = arg->src[idx] * 2;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->count, nullptr, kernel_body, arg);
}
```

**HIP**:
```cpp
__global__ void myKernel(int* src, int* dst, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx] * 2;
    }
}
// No main() needed - HIP runtime handles launch
```

### 7.3 Compilation Flow Comparison

#### Standard Vortex
```
┌──────────────┐                    ┌──────────────┐
│  main.cpp    │                    │ kernel.cpp   │
│  (Host)      │                    │  (Kernel)    │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       ↓ [g++]                             ↓ [llvm-vortex clang++]
       │                                   │ +vortex feature
       │                                   │ Links libvortex.a
┌──────┴───────┐                    ┌──────┴───────┐
│ host binary  │                    │ kernel.vxbin │
│  (x86 ELF)   │                    │ (RISC-V bin) │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │ Links libvortex.so                │ Uploaded at runtime
       └────────────┬──────────────────────┘
                    ↓
              vx_start(kernel)
```

#### HIP Integration
```
┌─────────────────────────────────────────┐
│            basic.hip                    │
│  (Unified host + kernel source)         │
└─────────────┬───────────────────────────┘
              │
              ↓ [Polygeist --cuda-lower]
              │
┌─────────────┴───────────────────────────┐
│         MLIR (GPU Dialect)              │
│                                         │
│  ┌─────────────┐   ┌──────────────┐   │
│  │ func.func   │   │ gpu.module   │   │
│  │ (Host)      │   │ (Kernel)     │   │
│  └─────┬───────┘   └──────┬───────┘   │
│        │                  │            │
└────────┼──────────────────┼────────────┘
         │                  │
         ↓                  ↓
  [ConvertGPUToVortex]  [ConvertGPUToVortex]
         │                  │
         ↓ LLVM dialect     ↓ LLVM dialect
         │ vx_* calls       │ vx_get_* TLS access
         │                  │
         ↓ [LLVM passes]    ↓ [LLVM passes]
         │                  │
┌────────┴────────┐  ┌──────┴────────┐
│  host binary    │  │ kernel.vxbin  │
│   (x86 ELF)     │  │ (RISC-V bin)  │
└────────┬────────┘  └──────┬────────┘
         │                  │
         │ Links libvortex.so│ Uploaded at runtime
         └─────────┬─────────┘
                   ↓
         vx_start(kernel) [generated by lowering]
```

### 7.4 Integration Strategy

Both approaches ultimately use the **same underlying runtime**:
- Same **libvortex.so** host runtime
- Same **libvortex.a** kernel library
- Same **.vxbin** binary format
- Same **vx_spawn** thread framework

**HIP integration goal**: Transform HIP API calls into equivalent vx_* calls at compile time, producing binaries identical to hand-written standard Vortex code.

---

## 4. COMPILER TOOLCHAIN INTEGRATION

### 4.1 Compiler Components and Locations

**LLVM-Vortex Backend:**
- **Location**: `llvm-vortex/`
- **Purpose**: LLVM compiler with Vortex-specific RISC-V target extensions
- **Base**: LLVM 14+ monorepo (llvm, clang, compiler-rt, etc.)
- **Target**: RISC-V with custom `+vortex` feature flag
- **Used by**: Standard Vortex compilation (direct), HIP integration (via MLIR)

**RISC-V GNU Toolchain:**
- **Location**: `$(TOOLDIR)/riscv32-gnu-toolchain` or `riscv64-gnu-toolchain`
- **Purpose**: GCC cross-compiler, binutils, and C library for final linking
- **Prefix**: `riscv32-unknown-elf` or `riscv64-unknown-elf`
- **Used by**: Standard Vortex (linking), HIP integration (linking)

**POCL (Portable OpenCL):**
- **Location**: `$(TOOLDIR)/pocl`
- **Purpose**: OpenCL runtime with Vortex device backend
- **Integration**: Uses LLVM-Vortex for code generation
- **Note**: Alternative to HIP for portable GPU programming

**Polygeist (HIP Integration Only):**
- **Location**: `Polygeist/`
- **Purpose**: HIP/CUDA to MLIR GPU dialect converter
- **Integration**: Frontend for HIP compilation pipeline
- **Note**: Not used by standard Vortex compilation

### 4.2 Environment Configuration

From `vortex/build/config.mk`:

```makefile
# Toolchain Locations
TOOLDIR ?= $(HOME)/tools
LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv$(XLEN)-gnu-toolchain
RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

# Vortex Specific
VORTEX_HOME ?= vortex
XLEN ?= 32  # 32-bit or 64-bit

# C Library
LIBC_VORTEX ?= $(TOOLDIR)/libc$(XLEN)
LIBCRT_VORTEX ?= $(TOOLDIR)/libcrt$(XLEN)
```

### 4.3 Toolchain Installation

**Installation Script**: `vortex/ci/toolchain_install.sh.in`

The script downloads and installs prebuilt toolchains:

```bash
# Install LLVM-Vortex
./toolchain_install.sh --llvm

# Install RISC-V GNU toolchain
./toolchain_install.sh --riscv32  # or --riscv64

# Install POCL
./toolchain_install.sh --pocl

# Install runtime libraries
./toolchain_install.sh --libcrt32 --libc32

# Install everything
./toolchain_install.sh --all
```

**Prebuilt Repository**: `https://github.com/vortexgpgpu/vortex-toolchain-prebuilt`

---

## 5. DETAILED COMPILATION PIPELINE (Standard Vortex)

### 5.1 Complete Flow

```
OpenCL/GPU Source (.cl)
         |
         v
   POCL Frontend
         |  (Uses Clang from LLVM-Vortex)
         v
   LLVM IR Generation
         |  (with OpenCL builtins)
         v
   LLVM Optimization Passes
         |  (Standard LLVM -O3)
         v
   Vortex-Specific Passes
         |  - Branch divergence handling
         |  - Vortex intrinsic lowering
         |  - Custom instruction selection
         v
   LLVM RISC-V Backend
         |  Target: riscv32/64 +vortex +zicond
         v
   RISC-V Assembly (.s)
         |
         v
   RISC-V GCC Assembler
         |
         v
   Object Files (.o)
         |
         v
   RISC-V GCC Linker
         |  Links with:
         |  - libvortex.a (GPU runtime)
         |  - libc.a, libm.a (standard libs)
         |  - libclang_rt.builtins-riscv32.a
         |  - Custom linker script (link32.ld)
         v
   ELF Binary
         |
         v
   vxbin.py Post-Processing
         |  - Extracts LOAD segments
         |  - Creates .vxbin format
         v
   Vortex Binary (.vxbin)
         |
         v
   Upload to Vortex Device
```

### 5.2 Compilation Flags

From `vortex/kernel/Makefile` and test Makefiles:

**Architecture Flags:**
```makefile
# 32-bit
CFLAGS += -march=rv32imaf -mabi=ilp32f

# 64-bit
CFLAGS += -march=rv64imafd -mabi=lp64d
```

**Vortex-Specific Flags:**
```makefile
# Enable Vortex target feature
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex

# Branch divergence handling (optional)
LLVM_CFLAGS += -mllvm -vortex-branch-divergence=0

# Disable loop idiom recognition (prevents memset/memcpy generation)
CFLAGS += -mllvm -disable-loop-idiom-all

# Code model and sysroot
CFLAGS += -mcmodel=medany
CFLAGS += --sysroot=$(RISCV_SYSROOT)
CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
```

**Optimization and Code Generation:**
```makefile
CFLAGS += -O3
CFLAGS += -fno-rtti -fno-exceptions
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += -nostartfiles -nostdlib  # For kernels
```

**Linker Flags:**
```makefile
LDFLAGS += -Wl,-Bstatic,--gc-sections  # Static linking, remove unused
LDFLAGS += -T$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld
LDFLAGS += --defsym=STARTUP_ADDR=$(STARTUP_ADDR)
LDFLAGS += $(VORTEX_HOME)/kernel/libvortex.a
LDFLAGS += -lm -lc
```

### 5.3 Kernel Library Build

**Location**: `vortex/kernel/`

**Makefile**: `vortex/kernel/Makefile`

**Build Process:**
```makefile
# Source files for libvortex.a
SRCS = vx_start.S       # Startup code (wspawn, TLS setup, main)
       vx_syscalls.c    # System call implementations
       vx_print.S       # Low-level print functions
       vx_print.c       # Printf implementation
       tinyprintf.c     # Lightweight printf
       vx_spawn.c       # Grid/block spawn framework
       vx_serial.S      # Serial I/O
       vx_perf.c        # Performance counters

# Compiler
CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar

# Build
$(PROJECT).a: $(OBJS)
    $(AR) rcs $@ $^
```

**Note**: Can also use LLVM toolchain (commented out in Makefile):
```makefile
#CC = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
#AR = $(LLVM_VORTEX)/bin/llvm-ar
```

---

## 3. VORTEX ISA AND PRIMITIVES

### 3.1 Base ISA

**RISC-V Standard Extensions:**
- **RV32IMAF** (32-bit) or **RV64IMAFD** (64-bit)
  - I: Integer base instructions
  - M: Integer multiply/divide
  - A: Atomic operations (optional, for synchronization)
  - F: Single-precision floating-point
  - D: Double-precision floating-point (64-bit only)

**Vortex Custom Extensions:**
- **+vortex**: Custom GPU instructions in RISC-V CUSTOM-0 opcode space (0x0B)
- **+zicond**: Conditional operations extension (RISC-V ratified)

### 3.2 Vortex GPU Instructions

**Header**: `vortex/kernel/include/vx_intrinsics.h`

**Thread Control:**
```c
void vx_tmc(int thread_mask);           // Thread mask control
void vx_tmc_zero();                     // Disable all threads
void vx_tmc_one();                      // Enable only thread 0

void vx_wspawn(int num_warps, void (*func_ptr)());  // Spawn warps

void vx_pred(int condition, int mask);  // Thread predication
int vx_split(int predicate);            // Divergence: save state
void vx_join(int stack_ptr);            // Convergence: restore state
```

**Synchronization:**
```c
void vx_barrier(int barrier_id, int num_warps);  // Warp barrier
void vx_fence();                                 // Memory fence
```

**Thread ID Queries (CSR reads):**
```c
int vx_thread_id();      // Thread within warp (CSR 0xCC0)
int vx_warp_id();        // Warp within core (CSR 0xCC1)
int vx_core_id();        // Core ID (CSR 0xCC2)
int vx_num_threads();    // Threads per warp (CSR 0xFC0)
int vx_num_warps();      // Warps per core (CSR 0xFC1)
int vx_num_cores();      // Total cores (CSR 0xFC2)
```

**Warp Collective Operations:**
```c
int vx_vote_all(int predicate);     // All threads true
int vx_vote_any(int predicate);     // Any thread true
int vx_vote_ballot(int predicate);  // Bitmask
int vx_shfl_up/down/bfly/idx(...);  // Warp shuffles
```

### 3.3 Assembly Encoding

All Vortex instructions use `.insn r` format with `RISCV_CUSTOM0 (0x0B)`:

```assembly
# Thread Mask Control
.insn r RISCV_CUSTOM0, 0, 0, x0, rs1, x0

# Warp Spawn
.insn r RISCV_CUSTOM0, 1, 0, x0, rs1, rs2

# Split/Join (divergence)
.insn r RISCV_CUSTOM0, 2, 0, rd, rs1, x0  # split
.insn r RISCV_CUSTOM0, 3, 0, x0, rs1, x0  # join

# Barrier
.insn r RISCV_CUSTOM0, 4, 0, x0, rs1, rs2

# Predication
.insn r RISCV_CUSTOM0, 5, 0, func, rs1, rs2  # func=0/1 for pred/pred_n
```

---

## 4. THREAD/WARP/CORE MODEL

### 4.1 Hierarchy

```
Vortex Device
  └── Clusters (NUM_CLUSTERS)
       └── Cores (NUM_CORES per cluster)
            └── Warps (NUM_WARPS per core, default: 4)
                 └── Threads (NUM_THREADS per warp, default: 4)
```

**Default Configuration**: 1 cluster, 1 core, 4 warps, 4 threads = 16 hardware threads

### 4.2 Mapping to CUDA/HIP

**Spawn API** (`vortex/kernel/include/vx_spawn.h`):

```c
typedef union { uint32_t x, y, z; } dim3_t;

extern __thread dim3_t blockIdx;    // Per-thread
extern __thread dim3_t threadIdx;   // Per-thread
extern dim3_t gridDim;              // Global
extern dim3_t blockDim;             // Global

int vx_spawn_threads(
    uint32_t dimension,
    const uint32_t* grid_dim,     // Number of blocks
    const uint32_t* block_dim,    // Threads per block
    vx_kernel_func_cb kernel_func,
    const void* arg
);
```

**Mapping Table:**

| HIP/CUDA | Vortex | Access Method |
|----------|--------|---------------|
| `gridDim.x` | `gridDim.x` | Global variable |
| `blockDim.x` | `blockDim.x` | Global variable |
| `blockIdx.x` | `blockIdx.x` | `__thread` TLS variable |
| `threadIdx.x` | `threadIdx.x` | `__thread` TLS variable |
| Warp ID | - | `vx_warp_id()` CSR read |
| Thread ID within warp | - | `vx_thread_id()` CSR read |
| `__syncthreads()` | `vx_barrier(__local_group_id, __warps_per_group)` | Intrinsic |

### 4.3 Spawn Implementation

**Source**: `vortex/kernel/src/vx_spawn.c`

The spawn framework distributes grid/blocks across Vortex's warp/thread hierarchy:

```c
// For each block in the grid:
//   Calculate blockIdx.{x,y,z}
//
//   If block has multiple threads:
//     Calculate warps_per_block = ceil(threads_per_block / threads_per_warp)
//     Spawn warps_per_block warps
//     Each warp calculates its threadIdx.{x,y,z}
//
//   If block has one thread:
//     One thread processes entire block
//     threadIdx = {0,0,0}
```

**Key Insight**: The spawn framework **automatically handles** the mapping from HIP's grid/block model to Vortex's warp/thread model. User kernels just read `blockIdx` and `threadIdx`.

---

## 5. MEMORY MODEL

### 5.1 Address Spaces

**Memory Map** (from `vortex/build/hw/VX_config.h`):

```
32-bit:
  IO_BASE_ADDR:        0x00000040
  USER_BASE_ADDR:      0x00010000  (User programs start here)
  STARTUP_ADDR:        0x80000000  (Kernel entry point)
  LMEM_BASE_ADDR:      0xFFFF0000  (Local/shared memory)
  STACK_BASE_ADDR:     0xFFFF0000  (Stack region)

64-bit:
  IO_BASE_ADDR:        0x000000040
  USER_BASE_ADDR:      0x000010000
  STARTUP_ADDR:        0x080000000
  LMEM_BASE_ADDR:      0x1FFFF0000
  STACK_BASE_ADDR:     0x1FFFF0000
```

**HIP/OpenCL Qualifiers:**

| Qualifier | Vortex Implementation |
|-----------|----------------------|
| `__global` / (default) | Standard memory (cached via L1/L2/L3) |
| `__shared__` / `__local` | LMEM (scratchpad per core) |
| `__constant__` | `.rodata` section (cached, read-only) |

**Local Memory (LMEM):**
- Base address: `CSR VX_CSR_LOCAL_MEM_BASE (0xFC3)`
- Size: Configurable, default 16 KB per core
- Allocation: `__local_mem(size)` macro offsets by `__local_group_id`

```c
#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)
```

### 5.2 Cache Hierarchy

```
Core
 ├── Icache (Instruction cache, 16KB default)
 └── Dcache (Data cache, 16KB default)
      └── L2 Cache (1MB default, shared across socket)
           └── L3 Cache (2MB default, shared across cluster)
                └── Main Memory
```

**Cache Properties:**
- Write-through with MSHR (Miss Status Holding Registers)
- Non-blocking pipeline
- Multi-bank parallelism for high throughput
- Coalescing in LSU (Load-Store Unit)

---

## 6. BINARY FORMAT AND LOADING

### 6.1 Vortex Binary Format (.vxbin)

**Tool**: `vortex/kernel/scripts/vxbin.py`

**Format:**
```
+---------------------------+
| min_vma (8 bytes, uint64) |  Minimum virtual address
+---------------------------+
| max_vma (8 bytes, uint64) |  Maximum virtual address
+---------------------------+
| Binary data (N bytes)     |  Raw binary from ELF LOAD segments
+---------------------------+
```

**Creation Process:**
```bash
# 1. Extract LOAD segments from ELF
readelf -l kernel.elf | grep LOAD

# 2. Determine VMA range (min_vma, max_vma)
# 3. Extract binary with objcopy
objcopy -O binary kernel.elf kernel.bin

# 4. Create .vxbin with header
vxbin.py kernel.elf kernel.vxbin
```

### 6.2 Linker Script

**Location**: `vortex/kernel/scripts/link32.ld` (or `link64.ld`)

**Key Sections:**
```ld
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = STARTUP_ADDR, LENGTH = ...
}

SECTIONS {
    .init    : { *(.init) }       # Initialization
    .text    : { *(.text*) }      # Code
    .rodata  : { *(.rodata*) }    # Read-only data
    .data    : { *(.data*) }      # Initialized data
    .bss     : { *(.bss*) }       # Uninitialized data
    .tdata   : { *(.tdata*) }     # Thread-local initialized
    .tbss    : { *(.tbss*) }      # Thread-local uninitialized
}
```

**Startup Address:**
- Defined via `--defsym=STARTUP_ADDR=0x80000000`
- Where `_start` is placed in memory

### 6.3 Runtime Loading

**API** (`vortex/runtime/include/vortex.h`):

```c
// Upload kernel binary to device
int vx_upload_kernel_file(vx_device_h device,
                          const char* filename,
                          vx_buffer_h* buffer);

// Start kernel execution
int vx_start(vx_device_h device,
            vx_buffer_h kernel,
            vx_buffer_h arguments);

// Wait for completion
int vx_ready_wait(vx_device_h device, uint64_t timeout);
```

---

## 7. COMPILER INTEGRATION WORKFLOW

### 7.1 Example: OpenCL Kernel Compilation

**Test Location**: `vortex/tests/opencl/vecadd/`

**Makefile** (`common.mk`):

```makefile
# Include Vortex configuration
include $(VORTEX_HOME)/build/config.mk

# Compile flags
VX_CFLAGS = -march=rv32imaf -mabi=ilp32f
VX_CFLAGS += -Xclang -target-feature -Xclang +vortex
VX_CFLAGS += -O3 -mcmodel=medany
VX_CFLAGS += --sysroot=$(RISCV_SYSROOT)
VX_CFLAGS += -fno-rtti -fno-exceptions
VX_CFLAGS += -nostartfiles -nostdlib

# Link flags
VX_LDFLAGS = -Wl,-Bstatic,--gc-sections
VX_LDFLAGS += -T$(VORTEX_HOME)/kernel/scripts/link32.ld
VX_LDFLAGS += $(VORTEX_HOME)/kernel/libvortex.a

# POCL environment for runtime compilation
POCL_CC_FLAGS = LLVM_PREFIX=$(LLVM_VORTEX)
POCL_CC_FLAGS += POCL_VORTEX_XLEN=32
POCL_CC_FLAGS += POCL_VORTEX_BINTOOL="$(VXBIN_TOOL)"
POCL_CC_FLAGS += POCL_VORTEX_CFLAGS="$(VX_CFLAGS)"
POCL_CC_FLAGS += POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)"
```

**Compilation at Runtime:**
1. Application calls `clBuildProgram()`
2. POCL Vortex backend invokes LLVM-Vortex
3. Compiles with `VX_CFLAGS`, links with `VX_LDFLAGS`
4. Runs `vxbin.py` to create `.vxbin`
5. Returns compiled kernel to application

### 7.2 HIP Integration Points

For HIP → Vortex compilation, integrate at these points:

**Option A: Runtime Compilation (like POCL)**
```
HIP Source → Polygeist → GPU Dialect IR → LLVM IR →
  LLVM-Vortex Backend → RISC-V Assembly → Link → .vxbin
```

**Option B: Ahead-of-Time Compilation**
```
HIP Source → Offline Compiler → .vxbin →
  Embed in application → Runtime loads .vxbin
```

**Recommended**: Hybrid approach
- Use Polygeist for GPU dialect generation (offline or runtime)
- Custom pass to lower GPU dialect → Vortex spawn API
- LLVM-Vortex backend for code generation
- Standard linking with `libvortex.a`

---

## 8. KEY INTEGRATION REQUIREMENTS FOR HIP

### 8.1 Compiler Modifications Needed

1. **HIP Intrinsic Lowering**:
   - Lower `threadIdx/blockIdx/blockDim/gridDim` to TLS reads or spawn variables
   - Convert `__syncthreads()` to `vx_barrier()`
   - Map `__shared__` to LMEM allocation

2. **Kernel Wrapping**:
   - Generate spawn framework wrapper (like `vx_spawn.c`)
   - Set up `blockIdx/threadIdx` calculation per iteration
   - Handle grid/block dimension loops

3. **Shared Memory Allocation**:
   - Calculate total `__shared__` memory per block
   - Generate `__local_mem(size)` allocation calls
   - Offset by `__local_group_id`

4. **Divergence Handling** (optional):
   - Insert `vx_split()/vx_join()` for divergent branches
   - Or rely on LLVM's existing infrastructure

### 8.2 Build System Integration

**Add to Vortex Makefiles**:

```makefile
# HIP Compilation Flags
HIP_CFLAGS = $(VX_CFLAGS)
HIP_CFLAGS += -I$(HIP_INCLUDE_PATH)
HIP_CFLAGS += --hip-device-lib-path=...

# HIP Lowering
HIP_LDFLAGS = $(VX_LDFLAGS)

# Compile HIP kernel
%.hip.o: %.hip
    $(LLVM_VORTEX)/bin/clang $(HIP_CFLAGS) -c $< -o $@

# Link
kernel.elf: kernel.hip.o
    $(LLVM_VORTEX)/bin/clang $(HIP_LDFLAGS) $^ -o $@

# Package
kernel.vxbin: kernel.elf
    $(VXBIN_TOOL) $< $@
```

### 8.3 Runtime Integration

**Update Vortex Runtime**:

```c
// New HIP-specific runtime API
int vx_hip_launch_kernel(
    vx_device_h device,
    vx_buffer_h kernel,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem
);
```

**Internally calls**:
```c
vx_spawn_threads(3, gridDim, blockDim, kernel_func, args);
```

---

## 9. SUMMARY AND RECOMMENDATIONS

### 9.1 Key Takeaways

1. **Vortex uses standard LLVM/RISC-V toolchain** with custom `+vortex` feature
2. **libvortex.a provides spawn framework** that handles grid/block mapping
3. **Binary format (.vxbin)** is simple: VMA header + raw binary
4. **POCL demonstrates complete integration** for OpenCL, can be model for HIP
5. **LLVM-Vortex backend is at** `llvm-vortex/`

### 9.2 HIP Compilation Strategy

**Phase 1: Leverage Polygeist for Parsing**
- Use Polygeist to parse HIP syntax → GPU dialect IR
- Manual observation of patterns (already done)

**Phase 2: Custom Lowering Pass**
- Create MLIR pass to lower GPU dialect → Vortex spawn API calls
- Map GPU operations to Vortex intrinsics
- Generate LLVM IR with Vortex-specific constructs

**Phase 3: Use LLVM-Vortex Backend**
- Feed LLVM IR to existing LLVM-Vortex backend
- Backend handles RISC-V code generation with `+vortex` features
- Link with `libvortex.a` using standard Vortex build process

**Phase 4: Runtime Integration**
- Create HIP runtime wrapper (`hip_runtime.h` → Vortex API)
- Use existing Vortex device management
- Upload .vxbin kernels using `vx_upload_kernel_file()`

This approach **reuses maximum infrastructure** while adding HIP-specific lowering at the right abstraction layer (MLIR/GPU dialect).

---

## File Reference Summary

**Compiler Toolchain:**
- `llvm-vortex/` - LLVM backend
- `vortex/build/config.mk` - Build configuration
- `vortex/ci/toolchain_install.sh.in` - Toolchain setup

**Kernel Library:**
- `vortex/kernel/Makefile` - libvortex.a build
- `vortex/kernel/src/vx_spawn.c` - Spawn framework
- `vortex/kernel/include/vx_intrinsics.h` - GPU primitives

**Linker and Binary:**
- `vortex/kernel/scripts/link32.ld` - Linker script
- `vortex/kernel/scripts/vxbin.py` - Binary packager

**Examples:**
- `vortex/tests/opencl/` - OpenCL test kernels
- `vortex/tests/opencl/common.mk` - Build example
