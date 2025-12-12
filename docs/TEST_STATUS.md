# HIP Test Status

This document tracks the status of HIP test cases and what is required for failing tests to pass.

## Test Results Summary

| Test | Compile | Run | Status |
|------|---------|-----|--------|
| basic.hip | PASS | PASS | **WORKING** |
| relu.hip | PASS | PASS | **WORKING** |
| vecadd.hip | PASS | PASS | **WORKING** |
| printf.hip | PASS | PASS | **WORKING** |
| demo.hip | PASS | FAIL | Runtime issue (wrong output) |
| cta.hip | PASS | FAIL | Runtime issue |
| fence.hip | FAIL | - | `__threadfence` not supported |
| diverge.hip | FAIL | - | `__device__` functions + `key_t` conflict |
| dogfood.hip | FAIL | - | Multiple kernels + missing math functions |
| dotproduct.hip | PASS | FAIL | Runtime issue |
| dropout.hip | FAIL | - | `__device__` functions |
| madmax.hip | FAIL | - | `__device__` functions |
| mstress.hip | FAIL | - | Missing atomic functions |
| io_addr.hip | PASS | FAIL | Runtime issue |
| sort.hip | PASS | FAIL | Runtime issue |
| sgemv.hip | PASS | FAIL | Runtime issue |
| stencil3d.hip | PASS | FAIL | Runtime issue |
| conv3.hip | PASS | FAIL | Runtime issue |
| sgemm.hip | PASS | FAIL | 2D thread blocks not supported |
| sgemm2.hip | PASS | FAIL | 2D thread blocks not supported |
| sgemm_tcu.hip | FAIL | - | Multiple kernels |

**Summary:** 4 passing, 10 compile + run but fail at runtime, 7 fail at compile time

## Failure Categories

### 1. Runtime Failures (Tests Compile Successfully)

**Affected tests:** demo, cta, dotproduct, io_addr, sort, sgemv, stencil3d, conv3

These tests compile and run but produce incorrect results. Common causes:
- Argument passing issues between host and device
- Thread indexing or synchronization issues
- Memory layout differences between host (64-bit) and device (32-bit)

**Investigation needed:**
- Verify kernel argument marshaling
- Check thread ID calculations
- Validate memory access patterns

---

### 2. `__device__` Functions Not Supported

**Affected tests:** diverge, dropout, madmax

**Error pattern:**
```
error: '__device__' does not name a type
```

**Cause:** The host compilation pass doesn't recognize `__device__` function declarations. These are GPU-only helper functions that should be stripped or handled during host compilation.

**Fix required:**
- Option A: Add `__device__` as a macro that expands to nothing during host compilation
- Option B: Strip `__device__` functions from the transformed source for host compilation

---

### 3. Missing HIP Intrinsics/Built-ins

**Affected tests:** fence (`__threadfence`), mstress (atomics), dogfood (math functions)

**Error patterns:**
```
error: use of undeclared identifier '__threadfence'
error: use of undeclared identifier 'sqrtf'
error: use of undeclared identifier 'atomicAdd'
```

**Fix required:**
- Implement missing intrinsics in the runtime or as RISC-V equivalents
- For math functions: ensure `-lm` is linked and math.h is included
- For atomics: implement via RISC-V atomic instructions

---

### 4. Multiple Kernels Not Supported

**Affected tests:** dogfood (22 kernels), sgemm_tcu (multiple kernels)

**Error pattern:**
```
error: 'kernel_iadd' was not declared in this scope
```

**Cause:** The current pipeline generates stubs for only one kernel. Tests with multiple `__global__` functions fail because other kernel calls in host code become undefined.

**Fix required:**
- Extend pipeline to detect and generate launchers for ALL kernels in a file
- May need to compile multiple `.vxbin` files or combine them

---

### 5. 2D Thread Blocks Not Supported

**Affected tests:** sgemm, sgemm2

**Error pattern:**
```
*** error: [0,0] expected=X, actual=0.000000
```

**Cause:** These tests use 2D thread blocks (`dim3 threads(tile_size, tile_size)`), but the current Vortex runtime only supports 1D thread indexing.

**Fix required:**
- Implement 2D/3D thread block support in Vortex spawning
- Map `threadIdx.y`, `threadIdx.z`, `blockIdx.y`, `blockIdx.z`

---

## Working Tests Analysis

The four working tests (basic, relu, vecadd, printf) share these characteristics:
- Single kernel per file
- 1D thread blocks only
- No `__device__` helper functions
- Simple memory access patterns (no shared memory)
- Standard kernel arguments (pointers + scalars)

## Recent Fixes

1. **Kernel Naming Mismatch (FIXED):** Updated `generate_host_stubs.py` to correctly extract kernel names from Polygeist-generated metadata, matching the launcher names produced by `inject_kernel_launchers.py`.

2. **MLIR Cache Behavior (FIXED):** Updated `compile_hip.sh` to prefer fresh MLIR generation over cached files, falling back to cache only if generation fails.

3. **Printf Support (FIXED):** Updated `GenerateVortexMain.cpp` in Polygeist to handle pointer-type synthetic arguments (like format strings) by looking up global string constants instead of defaulting to `i32` zero.

## Priority Fixes

1. **High Priority - Runtime Issues:** Investigate why tests that compile produce wrong results. This affects: demo, cta, dotproduct, io_addr, sort, sgemv, stencil3d, conv3 (8 tests)

2. **Medium Priority - `__device__` Functions:** Add support for `__device__` functions in host compilation. This would enable: diverge, dropout, madmax (3 tests)

3. **Medium Priority - Missing Intrinsics:** Implement `__threadfence`, atomics, and math functions. This would enable: fence, mstress, dogfood (3 tests)

4. **Medium Priority - 2D Blocks:** Implement 2D thread block support. This would enable: sgemm, sgemm2 (2 tests)

5. **Low Priority - Multiple Kernels:** Support multiple kernels per file. This would enable: dogfood, sgemm_tcu (2 tests)

## Environment

- Vortex simulator: simx
- Target: RISC-V 32-bit
- Thread configuration: 16 threads per block max
