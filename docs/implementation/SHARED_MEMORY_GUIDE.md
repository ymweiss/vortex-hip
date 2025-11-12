# Vortex HIP Shared Memory Implementation Guide

**Date:** 2025-11-09
**Status:** ✅ Fully Implemented and Tested

## Overview

This guide documents how to use shared memory in Vortex HIP kernels. Shared memory (also called local memory) is critical for performance optimization in GPU kernels, particularly for reduction and tiling patterns.

## Key Finding

**Vortex uses the `__local_mem(size)` macro for shared memory allocation**, NOT CUDA/HIP's `__shared__` array syntax.

## Implementation Pattern

### In Kernel Code

```cpp
#include <vx_spawn.h>

struct KernelArgs {
    // Runtime-provided fields (always first)
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;  // Shared memory size in bytes

    // User arguments
    TYPE* input;
    TYPE* output;
    uint32_t size;
} __attribute__((packed));

void kernel_body(KernelArgs* __UNIFORM__ args) {
    // Allocate shared memory using __local_mem macro
    auto shared_data = reinterpret_cast<TYPE*>(__local_mem(args->shared_mem));

    // Use shared memory
    int tid = threadIdx.x;
    shared_data[tid] = args->input[tid];

    __syncthreads();  // Synchronize before using shared data

    // Process shared data...
}

int main() {
    KernelArgs* args = (KernelArgs*)csr_read(VX_CSR_MSCRATCH);

    // IMPORTANT: Pass both grid_dim and block_dim
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

### In Host Code

```cpp
#include "vortex_hip_runtime.h"

// Calculate shared memory size
size_t threadsPerBlock = 256;
size_t sharedMemBytes = threadsPerBlock * sizeof(TYPE);

// Launch kernel with shared memory
void* args[] = {&d_input, &d_output, &size};
HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                          dim3(blocksPerGrid),
                          dim3(threadsPerBlock),
                          args,
                          sharedMemBytes,  // Pass shared memory size
                          nullptr));
```

## How It Works

### The `__local_mem()` Macro

From `vortex/kernel/include/vx_spawn.h`:

```c
#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)
```

This macro:
1. Reads the local memory base address from CSR register `VX_CSR_LOCAL_MEM_BASE`
2. Offsets by `__local_group_id * size` to get this block's allocation
3. Returns a pointer to block-local shared memory

Each thread block (work group) gets its own isolated shared memory region.

### Runtime Support

The Vortex HIP runtime (`vortex_hip_runtime.cpp`, lines 595-598) automatically:
1. Receives `sharedMemBytes` parameter from `hipLaunchKernel()`
2. Packs it into the argument buffer as a `uint64_t`
3. Places it in the kernel argument structure after grid_dim and block_dim

**No runtime modifications needed** - it already works!

### Memory Layout

Argument buffer layout for kernels with shared memory:

```
Offset  Size  Field
------  ----  -----
0       12    grid_dim[3]      (uint32_t[3])
12      12    block_dim[3]     (uint32_t[3])
24      8     shared_mem       (uint64_t)
32      ...   user arguments
```

## Common Patterns

### 1. Reduction (e.g., Dot Product)

```cpp
void kernel_body(Args* __UNIFORM__ args) {
    auto cache = reinterpret_cast<TYPE*>(__local_mem(args->shared_mem));
    int tid = threadIdx.x;

    // Calculate partial result
    TYPE temp = 0;
    for (int i = tid; i < args->size; i += blockDim.x) {
        temp += args->input[i];
    }
    cache[tid] = temp;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        args->output[blockIdx.x] = cache[0];
    }
}
```

**Launch:** Each block produces one partial result, final reduction on CPU.

### 2. Tiled Matrix Multiply

```cpp
void kernel_body(Args* __UNIFORM__ args) {
    // Allocate space for two tiles
    auto local_ptr = __local_mem(2 * TILE_SIZE * TILE_SIZE * sizeof(TYPE));
    auto tile_A = (TYPE*)local_ptr;
    auto tile_B = tile_A + TILE_SIZE * TILE_SIZE;

    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    TYPE sum = 0;

    // Loop over tiles
    for (int t = 0; t < args->size / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        tile_A[threadIdx.x * TILE_SIZE + threadIdx.y] =
            args->A[row * args->size + (t * TILE_SIZE + threadIdx.y)];
        tile_B[threadIdx.x * TILE_SIZE + threadIdx.y] =
            args->B[(t * TILE_SIZE + threadIdx.x) * args->size + col];

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.x * TILE_SIZE + k] *
                   tile_B[k * TILE_SIZE + threadIdx.y];
        }

        __syncthreads();
    }

    args->C[row * args->size + col] = sum;
}
```

## Important Notes

### ✅ Do This

1. **Use `__local_mem()` macro** for shared memory allocation
2. **Pass both grid_dim and block_dim** to `vx_spawn_threads()`
3. **Include shared_mem field** in kernel argument structure
4. **Synchronize with `__syncthreads()`** before/after shared memory access
5. **Cast to appropriate type** using `reinterpret_cast<TYPE*>()`

### ❌ Don't Do This

1. ~~Don't use `__shared__` array syntax~~ - This is CUDA/HIP specific, not Vortex
2. ~~Don't use `extern __shared__` pattern~~ - Not supported in Vortex
3. ~~Don't pass total thread count~~ - Must use grid_dim and block_dim separately
4. Don't skip `__syncthreads()` - Leads to race conditions

## Tested Examples

### dotproduct_test - ✅ PASSING

- **Pattern:** Reduction with shared memory
- **Tested sizes:** 16, 256, 1024 elements
- **Location:** `/home/yaakov/vortex_hip/tests/dotproduct_test/`
- **Key features:**
  - Two-stage reduction (per-block, then CPU)
  - Uses `__local_mem()` for cache allocation
  - Tree reduction pattern

**Test results:**
```
n=16:   GPU=252,   CPU=252   ✅
n=256:  GPU=5674,  CPU=5674  ✅
n=1024: GPU=21602, CPU=21602 ✅
```

## Reference

- **Vortex spawn header:** `vortex/kernel/include/vx_spawn.h`
- **Runtime implementation:** `runtime/src/vortex_hip_runtime.cpp` (lines 595-598)
- **Example kernels:**
  - `vortex/tests/regression/dotproduct/kernel.cpp`
  - `vortex/tests/regression/sgemm2/kernel.cpp`
  - `tests/dotproduct_test/kernel.cpp` (our implementation)

## Debugging Tips

1. **Check shared memory size:** Ensure it matches allocation in kernel
2. **Verify synchronization:** Missing `__syncthreads()` causes race conditions
3. **Inspect argument structure:** Use `args->shared_mem` to verify size passed
4. **Watch for bank conflicts:** In future optimizations, consider memory layout

## Next Steps

With shared memory working:
1. ✅ Dotproduct test complete
2. 🔜 Implement sgemm2_test (tiled matrix multiply)
3. 🔜 Optimize memory access patterns
4. 🔜 Add more advanced shared memory patterns

---

**Status:** Fully implemented and validated
**Last Updated:** 2025-11-09
