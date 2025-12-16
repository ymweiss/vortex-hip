# HIP Test Status

This document tracks the status of HIP test cases and what is required for failing tests to pass.

**Last Updated:** 2025-12-16

## Test Results Summary

Using `compile_hip_v2.sh` pipeline:

| Test | Compile | Status | Notes |
|------|---------|--------|-------|
| basic.hip | PASS | **COMPILES** | 1D kernel |
| conv3.hip | PASS | **COMPILES** | 2D convolution |
| cta.hip | PASS | **COMPILES** | 3D grid/block |
| demo.hip | PASS | **COMPILES** | Simple kernel |
| diverge.hip | FAIL | Missing `min`/`max` | Device math functions |
| dogfood.hip | FAIL | Ambiguous `pow` | Math function overload |
| dotproduct.hip | FAIL | Host header conflict | `__shared__` vs `extern` |
| dropout.hip | FAIL | Device function issue | `WangHash` not found |
| fence.hip | FAIL | Missing `__threadfence` | Memory fence intrinsic |
| io_addr.hip | PASS | **COMPILES** | Pointer arithmetic |
| madmax.hip | PASS | **COMPILES** | Device helper functions |
| mstress.hip | PASS | **COMPILES** | Memory stress test |
| printf.hip | PASS | **COMPILES** | Device printf |
| relu.hip | PASS | **COMPILES** | ReLU activation |
| sgemm.hip | PASS | **COMPILES** | 2D matrix multiply |
| sgemm2.hip | FAIL | Host header conflict | `__shared__` vs `extern` |
| sgemm_tcu.hip | PASS | **COMPILES** | TCU-style matrix multiply |
| sgemv.hip | PASS | **COMPILES** | Matrix-vector multiply |
| simple_malloc_test.hip | PASS | **COMPILES** | Simple malloc test |
| sort.hip | PASS | **COMPILES** | Bitonic sort |
| stencil3d.hip | PASS | **COMPILES** | 3D stencil |
| vecadd.hip | PASS | **COMPILES** | Vector addition |
| vecadd_v2.hip | PASS | **COMPILES** | Vector addition (v2 style) |

**Summary:** 17/23 compile successfully (74%), 6 fail at compile time

## Failure Categories

### 1. Device Math Functions Missing (High Priority)

**Affected tests:** diverge, dogfood

**Error patterns:**
```
error: use of undeclared identifier 'min'; did you mean 'std::min'?
error: call to 'pow' is ambiguous
```

**Cause:** Device-side math functions (`min`, `max`, `pow`, etc.) are not declared in the device headers. The CUDA frontend doesn't provide these automatically.

**Fix required:**
Add device math function declarations to `runtime/device/hip/hip_runtime.h`:
```cpp
__device__ inline int min(int a, int b) { return a < b ? a : b; }
__device__ inline int max(int a, int b) { return a > b ? a : b; }
__device__ float powf(float base, float exp);
__device__ float sqrtf(float x);
// etc.
```

---

### 2. Memory Fence Intrinsics Missing (High Priority)

**Affected tests:** fence

**Error pattern:**
```
error: use of undeclared identifier '__threadfence'
```

**Cause:** Memory fence intrinsics are not implemented. Vortex needs fence operations for memory ordering.

**Fix required:**
- Declare `__threadfence()`, `__threadfence_block()`, `__threadfence_system()` in device header
- Lower to Vortex fence operations in `ConvertGPUToVortex.cpp`
- Or implement as no-ops if Vortex memory model is already sequentially consistent

---

### 3. Device Function Attribute Handling (High Priority)

**Affected tests:** dropout

**Error pattern:**
```
error: no matching function for call to 'WangHash'
```

**Cause:** The `__device__` helper functions (`WangHash`, `RandomInt`, `RandomFloat`) aren't being parsed correctly. Function signature or attribute handling issue during CUDA compilation.

**Investigation needed:**
- Check if `__device__` functions are being correctly parsed by cgeist
- Verify function declarations match between call sites and definitions

---

### 4. Host Header `__shared__` Conflict (Medium Priority)

**Affected tests:** dotproduct, sgemm2

**Error pattern:**
```
error: 'static' specifier conflicts with 'extern'
```

**Cause:** In `runtime/host/hip/hip_runtime.h`, `__shared__` is defined as `static`. When STL headers like `<vector>` are included, this causes conflicts with `extern` declarations in the standard library.

**Fix required:**
- Change the `__shared__` macro approach in host header
- Option A: Use `thread_local` instead of `static`
- Option B: Undefine `__shared__` before including STL headers
- Option C: Use a different identifier that doesn't conflict

---

## Recently Fixed Issues

### Multidimensional Grid/Block Support (FIXED)
- **Previously:** 2D/3D grids caused runtime failures
- **Now:** `ConvertGPUToVortex.cpp` extracts dimension from grid sizes and sets `vortex.kernel_dimension` attribute
- **Result:** sgemm, conv3, stencil3d, cta now compile with correct dimension passed to `vx_spawn_threads`

### Device Printf Support (FIXED)
- Printf calls in kernels are lowered to `vx_printf`
- Format string handling works correctly

### Shared Memory Support (FIXED)
- `__shared__` variables lowered to Vortex local memory
- `__syncthreads()` lowered to barrier operations

### `__device__` Helper Functions (PARTIALLY FIXED)
- **madmax** now compiles (device functions work)
- **dropout** still fails (different issue - likely function signature mismatch)

### Low-Priority Test Fixes (FIXED - 2025-12-16)
- **sgemm_tcu** - Replaced `typeid` with macro stringification (avoids `<typeinfo>`)
- **simple_malloc_test** - Updated cstdlib stub with missing stdlib declarations; converted to `hipLaunchKernelGGL`
- **vecadd_v2** - Changed to use `<hip/hip_runtime.h>` for unified device/host compilation

---

## Priority Fixes for 100% Pass Rate

| Priority | Feature | Tests Fixed | Effort |
|----------|---------|-------------|--------|
| 1 | Device math functions (min/max/pow) | diverge, dogfood | Low |
| 2 | Memory fence intrinsics | fence | Medium |
| 3 | Device function debugging | dropout | Medium |
| 4 | Host `__shared__` macro fix | dotproduct, sgemm2 | Low |

**Current pass rate: 17/23 (74%)**
**Expected pass rate after all fixes: 23/23 (100%)**

---

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| 1D grid/block | **Working** | Default mode |
| 2D grid/block | **Working** | dimension=2 passed to vx_spawn_threads |
| 3D grid/block | **Working** | dimension=3 passed to vx_spawn_threads |
| `__syncthreads()` | **Working** | Lowered to barrier |
| `__shared__` memory | **Working** | Lowered to local memory |
| Device printf | **Working** | Lowered to vx_printf |
| `__device__` functions | **Partial** | Works for some tests |
| Math functions | **Missing** | Need device declarations |
| Memory fences | **Missing** | Need implementation |
| Atomics | **Untested** | May work via RISC-V atomics |

---

## Environment

- Pipeline: `compile_hip_v2.sh`
- Vortex simulator: simx
- Target: RISC-V 32-bit (rv32imaf)
- Polygeist: Custom fork with Vortex passes
- Thread configuration: Configurable (default 4 threads per warp)
