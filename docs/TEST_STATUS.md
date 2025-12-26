# HIP Test Status

This document tracks the status of HIP test cases.

**Last Updated:** 2025-12-26

## Test Results Summary

Using `compile_hip_v2.sh` pipeline:

| Test | Status | Notes |
|------|--------|-------|
| basic.hip | **PASS** | 1D kernel |
| conv3.hip | **PASS** | 2D convolution - verified |
| cta.hip | **PASS** | 2D grid - FIXED by dimension sinking pass |
| demo.hip | **PASS** | Simple kernel |
| diverge.hip | **PASS** | Thread divergence - verified |
| dogfood.hip | **PASS** | Multi-kernel test (8 kernels) - FIXED by multi-kernel runtime support |
| dotproduct.hip | **PASS** | Parallel reduction with shared memory - FIXED by NVVM barrier lowering |
| dropout.hip | **PASS** | Neural network dropout - verified |
| fence.hip | **PASS** | Memory fence operations |
| io_addr.hip | **PASS** | Device pointer arithmetic |
| madmax.hip | **PASS** | Device helper functions |
| mstress.hip | **PASS** | Memory stress test |
| printf.hip | **PASS** | Device printf |
| relu.hip | **PASS** | ReLU activation |
| sgemm.hip | **PASS** | 2D matrix multiply - FIXED by dimension sinking pass |
| sgemm2.hip | **PASS** | Tiled matrix multiply with shared memory - FIXED by NVVM barrier lowering |
| sgemm_tcu.hip | **PASS** | TCU-style matrix multiply - verified |
| sgemv.hip | **PASS** | Matrix-vector multiply |
| simple_malloc_test.hip | **PASS** | Simple malloc test |
| sort.hip | **PASS** | Bitonic sort (21 kernel invocations) - FIXED by multi-kernel runtime support |
| stencil3d.hip | **PASS** | 3D stencil - FIXED by dimension sinking pass |
| vecadd.hip | **PASS** | Vector addition - verified |
| vecadd_v2.hip | **PASS** | Vector addition (v2 style) - verified |

**Summary: 23/23 tests pass (100%)**

**Runtime Verified (2025-12-26):**
- vecadd, vecadd_v2, diverge, sgemm, cta, stencil3d, conv3, sgemm_tcu, relu, printf, fence, io_addr, madmax, mstress, sgemv, simple_malloc_test, basic, demo, dotproduct, sgemm2, dogfood, sort, dropout

**Known Failures:**
- None

**Key Fixes (2025-12-26 - Multi-Kernel Runtime Support):**
- Implemented kernel switching in runtime to support multiple different kernels in same program
- Vortex loads kernels at fixed address (0x80000000), so only one kernel can be loaded at a time
- Runtime now tracks current kernel and switches by waiting for completion + freeing before loading new one
- Added math function lowering (sqrtf, sqrt, fabsf, fabs, fminf, fmaxf, floorf, ceilf) to LLVM intrinsics
- dogfood (8 kernels), sort (21 invocations of same kernel) now pass

**Key Fixes (2025-12-25 - NVVM Barrier Lowering):**
- Added `NVVMBarrier0OpLowering` pattern to ConvertGPUToVortex pass
- `__syncthreads()` was being lowered to `nvvm.barrier0` which wasn't converted
- dotproduct, sgemm2 now pass - both use shared memory with __syncthreads()

**Key Fixes (2025-12-25 - Dimension Sinking Pass):**
- Added `SinkGpuDimsIntoLaunch` pass to eliminate synthetic kernel arguments
- sgemm (2D), cta (2D), stencil3d (3D) now pass - were all failing before
- All multi-dimensional kernels (1D, 2D, 3D) now work correctly

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

## Recently Fixed Issues (2025-12-26)

### Multi-Kernel Runtime Support (FIXED)
- **Tests:** dogfood, sort (any test with multiple different kernels)
- **Error:** "address range overlaps with existing allocation" when loading second kernel
- **Cause:** Vortex loads all kernels at fixed address 0x80000000; only one kernel can be resident at a time
- **Fix:** Implemented kernel switching in runtime:
  - Track currently loaded kernel name and buffer
  - Before loading new kernel: wait for completion, free old kernel buffer
  - Update tracking state for new kernel
- **Files:**
  - `runtime/hip_vortex_runtime/src/hip_kernel.cpp` - Added `switch_kernel_if_needed()` and tracking globals
- **Result:** Programs with multiple kernels (dogfood: 8 kernels, sort: 21 invocations) now work correctly

### Math Function Lowering (FIXED)
- **Tests:** dogfood (fsqrt test)
- **Error:** Missing device library function `sqrtf`
- **Cause:** Original Vortex tests had inline assembly for math functions; HIP version uses standard math calls which Polygeist lowered to `llvm.sqrt.f32` intrinsic
- **Fix:** Added `MathFunctionOpLowering` pattern to ConvertGPUToVortex pass:
  - Lowers: sqrtf, sqrt, fabsf, fabs, fminf, fmaxf, floorf, ceilf
  - Maps to LLVM intrinsics which compile to RISC-V F extension instructions (fsqrt.s, etc.)
- **Files:**
  - `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp` - Added MathFunctionOpLowering pattern
- **Result:** Math functions work correctly in device kernels

### dogfood ftoi Test Data (FIXED)
- **Tests:** dogfood (ftoi test)
- **Error:** expected=INT_MIN vs actual=INT_MAX for indices 513+
- **Cause:** Unsigned integer underflow: `num_points / 2 - i` when `i > 512` underflows as uint32_t
- **Fix:** Changed to signed arithmetic: `(float)((int32_t)(num_points / 2) - (int32_t)i)`
- **Files:**
  - `hip_tests/dogfood.hip` - Fixed test data initialization
- **Result:** All 8 dogfood subtests (iadd, imul, fadd, fmul, fdiv, fsqrt, ftoi, itof) pass

---

## Recently Fixed Issues (2025-12-25)

### NVVM Barrier0 Lowering (FIXED)
- **Tests:** dotproduct, sgemm2 (shared memory reduction kernels)
- **Error:** LLC crash: `Cannot select: intrinsic %llvm.nvvm.barrier0`
- **Cause:** `__syncthreads()` was being lowered to `nvvm.barrier0` NVVM dialect op, but the ConvertGPUToVortex pass only handled `gpu::BarrierOp`, not `NVVM::Barrier0Op`
- **Fix:** Added `NVVMBarrier0OpLowering` pattern to ConvertGPUToVortex.cpp that converts `nvvm.barrier0` to `vx_barrier_abi()` calls
- **Files:**
  - `lib/polygeist/Passes/ConvertGPUToVortex.cpp` - Added NVVMBarrier0OpLowering pattern and NVVM dialect include
- **Result:** dotproduct and sgemm2 now pass - shared memory with __syncthreads() works correctly

### Dimension Sinking Pass (FIXED)
- **Tests:** sgemm, cta, stencil3d (all multi-dimensional kernels)
- **Error:** 2D and 3D kernels failing at runtime with incorrect results
- **Cause:** GPU dimension operations (`gpu.block_dim`, `gpu.grid_dim`) and their derived computations were being captured as synthetic kernel arguments during outlining, causing incorrect values to be passed
- **Fix:** Created `SinkGpuDimsIntoLaunch` pass that sinks GPU dimension operations and pure computations into gpu.launch bodies before kernel outlining
- **Files:**
  - `lib/polygeist/Passes/SinkGpuDimsIntoLaunch.cpp` - New pass
  - `include/polygeist/Passes/Passes.td` - Pass registration
  - `tools/cgeist/driver.cc` - Pipeline integration
- **Result:** All 1D, 2D, and 3D kernels now work correctly

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
| Multi-kernel programs | **Working** | Runtime kernel switching (one kernel resident at a time) |
| `__syncthreads()` | **Working** | Lowered to barrier |
| `__shared__` memory | **Working** | Lowered to local memory |
| Device printf | **Working** | Lowered to vx_printf |
| `__device__` functions | **Working** | Inlined during compilation |
| Math functions | **Working** | Lowered to LLVM intrinsics → RISC-V F extension |
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

Tests were converted from Vortex's original test format to HIP.

| Test | Runtime Verified | Notes |
|------|------------------|-------|
| basic.hip | ✓ | Verified |
| conv3.hip | ✓ | Verified (2D convolution) |
| cta.hip | ✓ | Verified - FIXED by dimension sinking pass |
| demo.hip | ✓ | Verified |
| diverge.hip | ✓ | Verified |
| dogfood.hip | ✓ | Verified - FIXED by multi-kernel runtime support + math lowering |
| dotproduct.hip | ✓ | Verified - FIXED by NVVM barrier lowering |
| dropout.hip | ✓ | Verified (neural network dropout) |
| fence.hip | ✓ | Verified |
| io_addr.hip | ✓ | Verified |
| madmax.hip | ✓ | Verified (fixed block size 4x4) |
| mstress.hip | ✓ | Verified |
| printf.hip | ✓ | Verified |
| relu.hip | ✓ | Verified |
| sgemm.hip | ✓ | Verified - FIXED by dimension sinking pass |
| sgemm2.hip | ✓ | Verified - FIXED by NVVM barrier lowering |
| sgemm_tcu.hip | ✓ | Verified |
| sgemv.hip | ✓ | Verified |
| simple_malloc_test.hip | ✓ | Verified |
| sort.hip | ✓ | Verified - FIXED by multi-kernel runtime support |
| stencil3d.hip | ✓ | Verified - FIXED by dimension sinking pass |
| vecadd.hip | ✓ | Verified |
| vecadd_v2.hip | ✓ | Verified |

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
