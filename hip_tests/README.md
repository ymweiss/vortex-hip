# HIP Regression Tests

This directory contains HIP versions of the Vortex regression tests, converted for Phase 2 Polygeist compilation testing.

## Overview

These tests were converted from the original Vortex regression tests located in `vortex/tests/regression/`. Each test has been transformed from the Vortex-specific format (using `vx_spawn.h` and Vortex runtime API) to standard HIP format with `__global__` kernels and HIP runtime API calls.

## Converted Tests

### ✅ Basic Operations

| Test | Description | Key Features |
|------|-------------|--------------|
| **vecadd.hip** | Vector addition | Simple element-wise operation, basic thread indexing |
| **basic.hip** | Memory copy | Basic memory operations, thread-to-data mapping |
| **relu.hip** | ReLU activation | Conditional operation (max(0, x)) |

### ✅ Linear Algebra

| Test | Description | Key Features |
|------|-------------|--------------|
| **dotproduct.hip** | Dot product with reduction | Shared memory, parallel reduction, `__syncthreads()` |
| **sgemm.hip** | Matrix multiplication | 2D grid/block indexing, nested loops |

### 🔄 To Be Converted

| Test | Description | Expected Features |
|------|-------------|-------------------|
| **sgemm2** | Matrix multiply with tiling | Shared memory tiling, advanced blocking |
| **conv3** | 3D convolution | 3D indexing, sliding window |
| **fence** | Memory fence test | `__threadfence()`, memory ordering |
| **cta** | Cooperative thread arrays | Block-level coordination |
| **diverge** | Control flow divergence | Branch divergence patterns |
| **madmax** | Computational stress test | Heavy arithmetic operations |
| **mstress** | Memory stress test | Memory access patterns |
| **demo** | Comprehensive demo | Multiple kernel features |

## Test Structure

Each `.hip` file contains:
1. **Kernel function** (`__global__`): GPU device code
2. **Host code** (`main()`): Memory allocation, data transfer, kernel launch, verification
3. **Error checking**: `HIP_CHECK` macro for all HIP API calls
4. **Verification**: Comparison with CPU-computed reference results

## Compilation (Future - Phase 2)

These tests will be compiled using:

```bash
# Convert HIP to MLIR using Polygeist
cgeist --cuda-lower test.hip -S -o test.mlir

# Apply MLIR passes
polygeist-opt --some-passes test.mlir -o test_transformed.mlir

# Convert to Vortex LLVM (custom pass)
# GPUToVortexLLVM pass will convert GPU dialect to Vortex runtime calls

# Generate Vortex binary
llvm-vortex ... -o test.vxbin
```

## Key HIP Features Used

### Thread Indexing
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 2D Indexing (for matrices)
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Shared Memory
```cpp
extern __shared__ TYPE cache[];  // Dynamic shared memory
```

### Synchronization
```cpp
__syncthreads();  // Block-level barrier
```

### Kernel Launch
```cpp
hipLaunchKernelGGL(kernel, blocks, threads, shared_mem, stream, args...);
```

## Conversion Notes

### Changes from Vortex Format

1. **Kernel Declaration**:
   - Vortex: `void kernel_body(kernel_arg_t* arg)`
   - HIP: `__global__ void kernel(TYPE* ptr, ...)`

2. **Thread Indexing**:
   - Vortex: Uses `blockIdx`, `threadIdx` directly (as globals)
   - HIP: Uses `blockIdx`, `threadIdx` (built-in variables)

3. **Shared Memory**:
   - Vortex: `__local_mem(size)`
   - HIP: `extern __shared__ TYPE array[]`

4. **Synchronization**:
   - Vortex: `__syncthreads()` (same)
   - HIP: `__syncthreads()` (same)

5. **Host API**:
   - Vortex: `vx_mem_alloc()`, `vx_copy_to_dev()`, `vx_start()`, etc.
   - HIP: `hipMalloc()`, `hipMemcpy()`, `hipLaunchKernelGGL()`, etc.

## Phase 2 Testing Plan

1. ✅ **Phase 2A**: Evaluate Polygeist output on simple kernels (vecadd)
2. 🔄 **Phase 2B**: Test all converted kernels through Polygeist pipeline
3. ⏭️ **Phase 2C**: Implement custom GPUToVortexLLVM pass
4. ⏭️ **Phase 2D**: End-to-end testing with Vortex runtime

## Status Summary

- **Converted**: 5 tests (vecadd, basic, dotproduct, relu, sgemm)
- **Remaining**: ~15 tests
- **Next Priority**: sgemm2 (shared memory tiling), fence (synchronization), cta (cooperation)

---

**Last Updated**: 2025-11-14
**Phase**: 2A Complete, 2B In Progress
