# vx_spawn_threads Usage Guide

**Date:** 2025-11-09
**Criticality:** ⚠️ HIGH - Incorrect usage causes deadlocks

## Overview

The `vx_spawn_threads` function is used in every Vortex kernel's device `main()` to spawn the thread hierarchy. **Incorrect usage causes runtime deadlocks and stalls.**

## Critical Rule

**ALWAYS pass `args->grid_dim` and `args->block_dim` arrays to `vx_spawn_threads`.**

**NEVER calculate total thread count yourself.**

## Function Signature

From `vortex/kernel/include/vx_spawn.h`:

```c
int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t* block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg);
```

**Parameters:**
- `dimension`: Number of dimensions (1, 2, or 3)
- `grid_dim`: Pointer to grid dimensions array `[x, y, z]`
- `block_dim`: Pointer to block dimensions array `[x, y, z]`
- `kernel_func`: Kernel body function pointer
- `arg`: Kernel arguments

## Correct Usage Patterns

### 1D Grid (Most Common)

```cpp
// Kernel argument structure
struct MyKernelArgs {
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;
    // ... user arguments
} __attribute__((packed));

void kernel_body(MyKernelArgs* __UNIFORM__ args) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel logic
}

// ✅ CORRECT
int main() {
    MyKernelArgs* args = (MyKernelArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

**Examples:** vecadd_test, relu_test, dotproduct_test, fence_test

### 2D Grid (Matrix Operations)

```cpp
void kernel_body(MatrixArgs* __UNIFORM__ args) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // ... kernel logic
}

// ✅ CORRECT
int main() {
    MatrixArgs* args = (MatrixArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(2, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

**Examples:** sgemm_test, sgemm2_test (future)

### 3D Grid (Volume Operations)

```cpp
void kernel_body(VolumeArgs* __UNIFORM__ args) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // ... kernel logic
}

// ✅ CORRECT
int main() {
    VolumeArgs* args = (VolumeArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(3, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

**Examples:** stencil3d_test (future), conv3_test (future)

## Common Mistakes

### ❌ MISTAKE 1: Calculating Total Thread Count

```cpp
// ❌ WRONG - Causes deadlocks!
int main() {
    Args* args = (Args*)csr_read(VX_CSR_MSCRATCH);
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];
    return vx_spawn_threads(1, &num_threads, nullptr, kernel_body, args);
}
```

**Problem:**
- Passing `nullptr` for `block_dim` breaks thread indexing
- `blockIdx` and `threadIdx` will not be set correctly
- Can cause runtime to hang indefinitely

**Symptom:**
- Test hangs/stalls during execution
- No output after "wait for completion"

**Fix:**
```cpp
// ✅ CORRECT
return vx_spawn_threads(1, args->grid_dim, args->block_dim, kernel_body, args);
```

### ❌ MISTAKE 2: Wrong Dimension Count

```cpp
// ❌ WRONG - Using 1D for a 2D kernel
int main() {
    MatrixArgs* args = (MatrixArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, args->grid_dim, args->block_dim, kernel_body, args);
}

void kernel_body(MatrixArgs* args) {
    // Uses blockIdx.y and threadIdx.y - but dimension=1!
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Wrong!
}
```

**Problem:**
- Dimension must match how kernel uses block/thread indices
- Using `.y` or `.z` with dimension=1 gives incorrect values

**Fix:**
```cpp
// ✅ CORRECT
return vx_spawn_threads(2, args->grid_dim, args->block_dim, kernel_body, args);
```

### ❌ MISTAKE 3: Passing Wrong Arrays

```cpp
// ❌ WRONG - Creating local arrays
int main() {
    Args* args = (Args*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid[3] = {10, 1, 1};
    uint32_t block[3] = {256, 1, 1};
    return vx_spawn_threads(1, grid, block, kernel_body, args);
}
```

**Problem:**
- Runtime sets grid_dim and block_dim in args structure
- Ignoring those values breaks host-device coordination

**Fix:**
```cpp
// ✅ CORRECT
return vx_spawn_threads(1, args->grid_dim, args->block_dim, kernel_body, args);
```

## Why It Works This Way

### Thread Hierarchy Setup

`vx_spawn_threads` uses grid_dim and block_dim to:

1. **Set up global variables:**
   - `blockIdx` - Current block index
   - `threadIdx` - Current thread index within block
   - `gridDim` - Total blocks in grid
   - `blockDim` - Total threads per block

2. **Spawn correct number of threads:**
   - Total threads = `grid_dim[0] * grid_dim[1] * grid_dim[2] * block_dim[0] * block_dim[1] * block_dim[2]`

3. **Enable proper indexing:**
   - Without block_dim, `blockIdx` cannot be calculated correctly
   - Without grid_dim, `gridDim` is not set for kernel use

### Memory Layout in Args

The runtime packs these values at the start of the argument structure:

```
Offset  Size  Field
------  ----  -----
0       12    grid_dim[3]      (uint32_t[3])
12      12    block_dim[3]     (uint32_t[3])
24      8     shared_mem       (uint64_t)
32      ...   user arguments
```

## Real-World Examples

### From fence_test (Before Fix - DEADLOCK)

```cpp
// ❌ CAUSED DEADLOCK
int main() {
    FenceArgs* args = (FenceArgs*)csr_read(VX_CSR_MSCRATCH);
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];
    return vx_spawn_threads(1, &num_threads, nullptr, kernel_body, args);
}

void kernel_body(FenceArgs* args) {
    vx_barrier(0, args->grid_dim[0]);  // Also wrong!
}
```

**Result:** Test hung indefinitely at "wait for completion"

### From fence_test (After Fix - PASSING)

```cpp
// ✅ FIXED - PASSES
int main() {
    FenceArgs* args = (FenceArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, args->grid_dim, args->block_dim, kernel_body, args);
}

void kernel_body(FenceArgs* args) {
    // Removed incorrect barrier - memory ordering handled by runtime
}
```

**Result:** Test passes ✅

### From dotproduct_test (Correct from Start)

```cpp
// ✅ CORRECT
int main() {
    DotproductArgs* args = (DotproductArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, args->grid_dim, args->block_dim, kernel_body, args);
}
```

**Result:** Test passes ✅

## Debugging Checklist

If your test hangs or produces incorrect results:

- [ ] Are you passing `args->grid_dim` (not a local variable)?
- [ ] Are you passing `args->block_dim` (not `nullptr`)?
- [ ] Does `dimension` match how kernel uses indices (1D, 2D, or 3D)?
- [ ] Are you using the correct vx_spawn_threads signature?
- [ ] Did you cast kernel_body to `(vx_kernel_func_cb)`?

## Quick Reference Card

```cpp
// Template for ALL Vortex kernels

struct KernelArgs {
    // ALWAYS include these first three fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;
    // ... your arguments
} __attribute__((packed));

void kernel_body(KernelArgs* __UNIFORM__ args) {
    // Your kernel code
}

int main() {
    KernelArgs* args = (KernelArgs*)csr_read(VX_CSR_MSCRATCH);

    // Choose ONE of these patterns:

    // 1D kernels (vecadd, relu, dotproduct, etc.)
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);

    // 2D kernels (sgemm, sgemm2, etc.)
    return vx_spawn_threads(2, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);

    // 3D kernels (conv3, stencil3d, etc.)
    return vx_spawn_threads(3, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

## Summary

✅ **DO:**
- Use `args->grid_dim` and `args->block_dim` directly
- Match dimension to kernel's index usage
- Keep argument structure layout consistent

❌ **DON'T:**
- Calculate total thread count yourself
- Pass `nullptr` for block_dim
- Create local grid/block arrays
- Ignore the values in args structure

---

**References:**
- `vortex/kernel/include/vx_spawn.h` - Function definition
- `runtime/src/vortex_hip_runtime.cpp` - How runtime sets up args
- `tests/fence_test/kernel.cpp` - Example of fix
- `tests/dotproduct_test/kernel.cpp` - Correct implementation

**Last Updated:** 2025-11-09
