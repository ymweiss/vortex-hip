# HIP Test Status

This document tracks the status of HIP test cases.

**Last Updated:** 2025-12-21

## Test Results Summary

Using `compile_hip_v2.sh` pipeline:

| Test | Status | Notes |
|------|--------|-------|
| basic.hip | **PASS** | 1D kernel |
| conv3.hip | **PASS** | 2D convolution |
| cta.hip | **PASS** | 3D grid/block |
| demo.hip | **PASS** | Simple kernel |
| diverge.hip | **PASS** | Thread divergence (constant args folded) |
| dogfood.hip | **PASS** | Arithmetic operations |
| dotproduct.hip | **PASS** | Shared memory reduction |
| dropout.hip | **PASS** | Neural network dropout |
| fence.hip | **PASS** | Memory fence operations |
| io_addr.hip | **PASS** | Device pointer arithmetic |
| madmax.hip | **PASS** | Device helper functions |
| mstress.hip | **PASS** | Memory stress test |
| printf.hip | **PASS** | Device printf |
| relu.hip | **PASS** | ReLU activation |
| sgemm.hip | **PASS** | 2D matrix multiply |
| sgemm2.hip | **PASS** | 2D matrix multiply (tiled) |
| sgemm_tcu.hip | **PASS** | TCU-style matrix multiply |
| sgemv.hip | **PASS** | Matrix-vector multiply |
| simple_malloc_test.hip | **PASS** | Simple malloc test |
| sort.hip | **PASS** | Bitonic sort |
| stencil3d.hip | **PASS** | 3D stencil |
| vecadd.hip | **PASS** | Vector addition |
| vecadd_v2.hip | **PASS** | Vector addition (v2 style) |

**Summary: 23/23 compile successfully (100%)**

**Note:** Compilation tests only. SimX runtime validation pending.

---

## Quick Start

### Compiling a Single Test

```bash
# Simple usage (output goes to same directory as input)
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip

# With custom output location (directory must exist)
mkdir -p build_output
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip -o build_output/vecadd

# With verbose output
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip --verbose

# Keep intermediate files for debugging
./scripts/compile_hip_v2.sh hip_tests/vecadd.hip --keep-temps
```

### Running a Compiled Test

```bash
# Run on Vortex simulator
VORTEX_HOME=/path/to/vortex ./hip_tests/vecadd
```

---

## Recently Fixed Issues (2025-12-21)

### Host-Side Constant Folding Mismatch (FIXED)
- **Tests:** diverge
- **Error:** Kernel crash due to argument count mismatch
- **Cause:** MLIR's constant folding removed compile-time constant kernel arguments (e.g., `samples=10`), but host stub still sent the original 4 args
- **Fix:** Added `ConstantArgumentAnalyzer` to HIPSourceTransform that detects constant args at launch sites and excludes them from wrapper/stub generation, matching MLIR's behavior

### Device Address Handling (FIXED)
- **Tests:** io_addr
- **Error:** Test used 64-bit host pointer values instead of 32-bit device addresses
- **Fix:** Use `hip_ptr_to_device_addr()` runtime function to get actual device addresses for device pointer arithmetic

### Kernel Name Extraction for uint64_t Pointers (FIXED)
- **Tests:** io_addr
- **Error:** Kernel name extracted as `io_addr_kernelPmPjj` instead of `io_addr_kernel`
- **Fix:** Added `Pm` (pointer to unsigned long) and other pointer type suffixes to `extractKernelNameFromWrapper()` in ReorderGPUKernelArgs pass

---

## Recently Fixed Issues (2025-12-16)

### Host `__shared__` Macro Conflict (FIXED)
- **Tests:** dotproduct, sgemm2
- **Error:** `'static' specifier conflicts with 'extern'`
- **Fix:** Changed `__shared__` macro to use `__attribute__((weak))` in host header

### Device Math Functions (FIXED)
- **Tests:** diverge, dogfood
- **Error:** `no member named 'min' in namespace 'std'`
- **Fix:** Added `<algorithm>` device stub with `__host__ __device__` min/max; added device math declarations

### Memory Fence Intrinsics (FIXED)
- **Tests:** fence
- **Error:** `'__threadfence' does not reference a valid function`
- **Fix:** Changed from extern declaration to inline no-op implementations

### Device Function Handling (FIXED)
- **Tests:** dropout
- **Error:** `no matching function for call to 'WangHash'`
- **Fix:** Changed device-only functions to `__host__ __device__` for functions called from both contexts

### Template/Lambda Complexity (FIXED)
- **Tests:** dogfood
- **Error:** MLIR `func.call` result type mismatch
- **Fix:** Rewrote test to use simple inline kernels without templates or lambdas

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
| `__device__` functions | **Working** | Inlined during compilation |
| Math functions | **Working** | Device stubs + declarations |
| Memory fences | **Working** | Inline no-ops (TODO: proper pass lowering) |
| Constant arg folding | **Working** | Host-side analysis matches MLIR folding |
| Device addresses | **Working** | `hip_ptr_to_device_addr()` for pointer arithmetic |
| Atomics | **Untested** | May work via RISC-V atomics |

---

## Environment

- Pipeline: `compile_hip_v2.sh`
- Vortex simulator: simx
- Target: RISC-V 32-bit (rv32imaf)
- Polygeist: Custom fork with Vortex passes
- Thread configuration: Configurable (default 4 threads per warp)

---

## Future Work

- Add proper memory fence lowering to MLIR pass (ConvertGPUToVortex) instead of inline no-ops
- Ensure device/host splitting removes unused kernel artifacts from host binaries
- Test atomic operations
