# GPU Dialect IR Format - Manual Observation Guide

## Overview

This document describes the GPU dialect IR format emitted by Polygeist when converting HIP/CUDA code. This IR serves as the intermediate representation that will be translated to Vortex ISA.

## Generation

Use the provided script to generate GPU dialect IR:

```bash
./scripts/polygeist/hip-to-gpu-dialect.sh <input.cu/.hip> [output.mlir]
```

Example:
```bash
./scripts/polygeist/hip-to-gpu-dialect.sh hip_tests/simple_kernel_with_host.cu
```

## Example GPU Dialect IR

For the simple kernel:
```cuda
__global__ void simple_add(int* data, int value) {
    int idx = threadIdx.x;
    data[idx] += value;
}

void launch_kernel(int* data, int value, int num_threads) {
    simple_add<<<1, num_threads>>>(data, value);
}
```

Polygeist generates:

```mlir
module attributes {
  gpu.container_module,
  llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu",
  polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
  polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda"
} {
  // GPU kernel module
  gpu.module @__polygeist_gpu_module {
    gpu.func @_Z13launch_kernelPiii_kernel94555991377168(%arg0: memref<?xi32>, %arg1: i32)
      kernel
      attributes {gpu.known_grid_size = array<i32: 1, 1, 1>}
    {
      %0 = gpu.thread_id  x                        // Get thread ID in X dimension
      %1 = memref.load %arg0[%0] : memref<?xi32>   // Load data[threadIdx.x]
      %2 = arith.addi %1, %arg1 : i32              // Add value
      memref.store %2, %arg0[%0] : memref<?xi32>   // Store back
      gpu.return
    }
  }

  // Host function that launches kernel
  func.func @_Z13launch_kernelPiii(%arg0: memref<?xi32>, %arg1: i32, %arg2: i32)
    attributes {llvm.linkage = #llvm.linkage<external>}
  {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index

    // Launch kernel: 1 block, %arg2 threads
    gpu.launch_func @__polygeist_gpu_module::@_Z13launch_kernelPiii_kernel94555991377168
      blocks in (%c1, %c1, %c1)
      threads in (%0, %c1, %c1)
      args(%arg0 : memref<?xi32>, %arg1 : i32)
    return
  }
}
```

## Key Components

### 1. Module Attributes

- **`gpu.container_module`**: Marks module as containing GPU code
- **`llvm.target_triple`**: Host architecture (x86_64)
- **`polygeist.gpu_module.llvm.target_triple`**: GPU target (nvptx64-nvidia-cuda)
  - **For Vortex**: This should be changed to Vortex ISA target

### 2. GPU Module (`gpu.module`)

Contains GPU kernel functions. Key elements:

- **`@__polygeist_gpu_module`**: Module name
- **`gpu.func`**: GPU kernel function
  - **`kernel` attribute**: Marks as kernel (vs device function)
  - **`gpu.known_grid_size`**: Grid dimensions if known at compile time

### 3. GPU Operations

#### Thread Identification
```mlir
%0 = gpu.thread_id x      // threadIdx.x
%1 = gpu.thread_id y      // threadIdx.y
%2 = gpu.thread_id z      // threadIdx.z
%3 = gpu.block_id x       // blockIdx.x
%4 = gpu.block_dim x      // blockDim.x
%5 = gpu.grid_dim x       // gridDim.x
```

#### Memory Operations
```mlir
%val = memref.load %ptr[%idx] : memref<?xi32>    // Load from memory
memref.store %val, %ptr[%idx] : memref<?xi32>    // Store to memory
```

#### Arithmetic Operations
```mlir
%sum = arith.addi %a, %b : i32      // Integer addition
%prod = arith.muli %a, %b : i32     // Integer multiplication
%fsum = arith.addf %a, %b : f32     // Float addition
```

#### Synchronization
```mlir
gpu.barrier    // __syncthreads()
```

### 4. Kernel Launch (`gpu.launch_func`)

```mlir
gpu.launch_func @module::@kernel_name
  blocks in (%grid_x, %grid_y, %grid_z)
  threads in (%block_x, %block_y, %block_z)
  args(%arg0 : type, %arg1 : type, ...)
```

- **blocks in**: Grid dimensions (number of blocks)
- **threads in**: Block dimensions (threads per block)
- **args**: Kernel arguments

## Mapping to Vortex

### Thread Model Mapping

| CUDA/HIP          | GPU Dialect      | Vortex Mapping (TBD) |
|-------------------|------------------|----------------------|
| `threadIdx.x`     | `gpu.thread_id x`| Vortex thread ID     |
| `blockIdx.x`      | `gpu.block_id x` | Vortex warp ID?      |
| `blockDim.x`      | `gpu.block_dim x`| Vortex threads/warp  |
| `gridDim.x`       | `gpu.grid_dim x` | Vortex num warps     |
| `__syncthreads()` | `gpu.barrier`    | Vortex barrier instr |

### Memory Model Mapping

| CUDA/HIP          | GPU Dialect      | Vortex Mapping (TBD) |
|-------------------|------------------|----------------------|
| Global memory     | `memref<?xT>`    | Vortex global mem    |
| Shared memory     | `memref<?xT, 3>` | Vortex shared mem    |
| Local/private     | SSA values       | Vortex registers     |

Address space encoding:
- 0: Generic
- 1: Global
- 3: Shared
- 5: Private/local

## Next Steps for Vortex Translation

### Phase 1: Understand Vortex ISA
- [ ] Document Vortex thread model (warps, threads, barriers)
- [ ] Document Vortex memory hierarchy (global, shared, registers)
- [ ] Document Vortex instruction format

### Phase 2: Create Translation Tool
- [ ] Parse GPU dialect MLIR
- [ ] Map GPU operations to Vortex instructions
- [ ] Generate Vortex assembly/IR

### Phase 3: Integration
- [ ] Create `SerializeToVortex` pass in Polygeist
- [ ] Or create standalone `mlir-to-vortex` tool

## Additional Examples Needed

For comprehensive understanding, generate GPU dialect IR for:

1. **2D/3D kernels**: Use `threadIdx.y/z`, `blockIdx.y/z`
2. **Shared memory**: Use `__shared__` arrays
3. **Synchronization**: Use `__syncthreads()`
4. **Atomics**: Use `atomicAdd`, etc.
5. **Math functions**: Use `sin`, `cos`, `sqrt`, etc.

## References

- MLIR GPU Dialect: https://mlir.llvm.org/docs/Dialects/GPU/
- Polygeist Documentation: Polygeist/README.md
- Script location: `scripts/polygeist/hip-to-gpu-dialect.sh`
