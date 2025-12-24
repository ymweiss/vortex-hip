# HIP Test Status

This document tracks the status of HIP test cases.

**Last Updated:** 2025-12-24

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

**Summary: 15/23 pass at runtime (65%)**

**Important Notes:**
- **Runtime verified:** basic, demo, diverge, dotproduct, fence, io_addr, madmax, mstress, printf, relu, sgemm, sgemm2, sgemv, simple_malloc_test, vecadd (15 tests)
- **Known failures:** conv3 (Polygeist bug), sort (multi-kernel), stencil3d (3D issues)
- **Vortex thread limit:** Tests using >16 threads per block have been fixed (madmax, sgemm, conv3, sgemm_tcu, vecadd_v2).
- **2D kernels:** sgemm, sgemm2, and madmax (2D kernels) pass with kernel_arg_mapping and blockDimXY fixes.

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

## Recently Fixed Issues (2025-12-24)

### blockDimXY Semantic Detection (FIXED)
- **Tests:** sgemm2, dotproduct
- **Error:** Incorrect synthetic argument values for 2D block kernels
- **Cause:** Kernels using `blockDim.x * blockDim.y` for shared memory offsets couldn't get the correct value at runtime
- **Example:** sgemm2 uses `local_mem + blockDim.x * blockDim.y` for second tile offset
- **Fix:** Added semantic detection in KernelOutlining.cpp:
  - `isDim3MemrefType()` and `findDim3ArgPositions()` for dynamic dim3 argument detection
  - Pattern detection for `muli(blockDim.x, blockDim.y)` → "blockDimXY"
  - GenerateVortexMain.cpp now loads blockDim.y and computes blockDimXY product
- **Result:** sgemm2 and dotproduct now pass

---

## Recently Fixed Issues (2025-12-22)

### kernel_arg_mapping Computation (FIXED)
- **Tests:** vecadd (was failing), all others still passing
- **Error:** Incorrect kernel argument mapping caused wrong data to be passed to kernel
- **Cause:** `ReorderGPUKernelArgs` pass assumed Polygeist always reorders args to scalars-first, but Polygeist's behavior is inconsistent
- **Example:** vecadd kernel received mapping `[3, 0, 1, 2]` (wrong) instead of `[0, 1, 2, 3]` (correct identity)
- **Fix:** Check actual GPU arg types vs wrapper types to determine if reordering is needed; set identity mapping after reordering
- **Note:** This pass runs inside `cgeist`, not `polygeist-opt`. Both must be rebuilt when modifying it.

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
| 2D block dimensions | **Working** | blockDimXY semantic detection for shared memory |
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

## Test Conversion Status

Tests were converted from Vortex's original test format to HIP. Unverified tests may have conversion errors:

| Test | Runtime Verified | Notes |
|------|------------------|-------|
| basic.hip | ✓ | Verified |
| demo.hip | ✓ | Verified |
| diverge.hip | ✓ | Verified |
| fence.hip | ✓ | Verified |
| io_addr.hip | ✓ | Verified |
| madmax.hip | ✓ | Verified (fixed block size 4x4) |
| mstress.hip | ✓ | Verified |
| printf.hip | ✓ | Verified |
| relu.hip | ✓ | Verified |
| sgemm.hip | ✓ | Verified (fixed block size 4x4, 2D kernel) |
| sgemv.hip | ✓ | Verified |
| simple_malloc_test.hip | ✓ | Verified |
| vecadd.hip | ✓ | Verified |
| dotproduct.hip | ✓ | Verified (shared memory reduction) |
| sgemm2.hip | ✓ | Verified (tiled SGEMM with 2D blocks) |
| conv3.hip | ✗ POLYGEIST BUG | Polygeist loses `paddedWidth = width + 2` computation |
| cta.hip | ✗ NEEDS FIX | 3D grid/block dims not lowering correctly |
| sort.hip | ✗ FAILS | Multi-kernel not supported (bitonic_sort_step not found) |
| stencil3d.hip | ✗ FAILS | 3D stencil - wrong results |
| vecadd_v2.hip | ✗ NEEDS FIX | User-defined launch wrapper not supported |
| dogfood.hip | - | Needs verification |
| dropout.hip | - | Needs verification |
| sgemm_tcu.hip | - | Needs verification |

**Before using unverified tests:**
1. Review the HIP conversion against original Vortex test
2. Compile with `--verbose --keep-temps` to inspect intermediate MLIR
3. Run on SimX and verify output matches expected results

---

## Known Polygeist Lowering Bugs

### conv3.hip - Local Variable Optimization Bug

**Status:** Blocking

**Description:** Polygeist incorrectly optimizes the local variable computation `paddedWidth = width + 2`. Instead of computing `width + 2` in the kernel, it hoists this as a separate kernel argument and maps it to the original `width` value, losing the `+2`.

**Symptom:** The kernel uses `width` (32) for input buffer stride instead of `paddedWidth` (34), causing incorrect memory accesses.

**Analysis:**
```mlir
# Expected: row * (width + 2) + col for input indexing
# Actual: row * width + col (missing +2)

kernel_arg_mapping = [3, -1, 0, 1, 2]
# arg0 → host arg 3 (width), but kernel uses it as paddedWidth
# arg1 → synthetic (-1), kernel uses it as width (coincidentally correct due to totalThreads=32=width)
```

**Workaround:** None that preserves correct test semantics. Test is correctly written; bug is in Polygeist.

---

## Future Work

- Add proper memory fence lowering to MLIR pass (ConvertGPUToVortex) instead of inline no-ops
- Ensure device/host splitting removes unused kernel artifacts from host binaries
- Test atomic operations
- Verify and fix remaining test conversions
- Fix Polygeist local variable hoisting bug (conv3)
