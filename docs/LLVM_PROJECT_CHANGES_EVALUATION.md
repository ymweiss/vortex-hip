# LLVM-Project Changes Evaluation

**Date:** 2025-12-25
**Evaluator:** Claude Code Assistant
**Context:** Post-Polygeist refactoring (SinkGpuDimsIntoLaunch pass, NVVM barrier lowering)

## Overview

This document evaluates the changes made to the `llvm-project` submodule within Polygeist to determine which are still necessary after the Polygeist refactoring work.

The llvm-project submodule contains 5 custom commits on top of upstream LLVM:

| Commit | Description | Files Changed |
|--------|-------------|---------------|
| `b67e4fa` | blockDimXY semantic detection | KernelOutlining.cpp |
| `8fb468b` | Synthetic semantic inference | KernelOutlining.cpp |
| `5b992e1` | Vortex metadata propagation | KernelOutlining.cpp |
| `ecdd57a` | kernel_arg_mapping attribute | KernelOutlining.cpp |
| `26eb428` | Vararg support (upstream) | Multiple LLVM files |

Additionally, there is 1 uncommitted change:
- Removed `polygeist.launch_wrapper` skip logic in KernelOutlining.cpp

---

## Detailed Analysis

### 1. kernel_arg_mapping Attribute (`ecdd57a`)

**Purpose:** Trace kernel arguments back to function parameters and record the mapping as an attribute on `gpu.func`.

**Changes Made:**
- Added `traceToFunctionArg()` - traces values through casts to function args
- Added `sortOperandsByFunctionOrder()` - sorts kernel operands by function arg order
- Sets `kernel_arg_mapping` attribute on outlined `gpu.func`

**Still Needed:** ✅ **YES**

**Reason:** This is fundamental infrastructure for the Vortex pipeline. The `kernel_arg_mapping` attribute is used by:
- `ReorderGPUKernelArgs` pass to determine correct argument order
- `GenerateVortexMain` pass to marshal arguments correctly
- Host stubs to pass arguments in the right order

Without this, the host-device argument ordering would be incorrect.

---

### 2. Vortex Metadata Propagation (`5b992e1`)

**Purpose:** Copy Vortex-specific attributes from `gpu.launch` to outlined `gpu.func`.

**Changes Made:**
- Copies `vortex.kernel_args`, `vortex.kernel_args_size`, `vortex.kernel_name` from launch op to func

**Still Needed:** ✅ **YES**

**Reason:** These attributes contain critical metadata:
- `vortex.kernel_name` - used to name the kernel binary correctly
- `vortex.kernel_args` - argument type information for marshaling
- `vortex.kernel_args_size` - total argument buffer size

The HIP source transformation sets these on `gpu.launch`, and they must survive outlining.

---

### 3. Synthetic Semantic Inference (`8fb468b`)

**Purpose:** Infer semantic meaning of synthetic kernel arguments (those not traceable to function args).

**Changes Made:**
- Added `inferSyntheticSemantic()` - detects patterns like:
  - `gpu.block_dim` / `gpu.grid_dim` operations
  - `muli(gridDim, blockDim)` → "totalThreads"
  - `divui(blockDim, 2)` → "blockDim/2"
- Sets `vortex.synthetic_semantic` attribute on synthetic args

**Still Needed:** ⚠️ **PARTIALLY**

**Reason:** The `SinkGpuDimsIntoLaunch` pass we created **eliminates most synthetic arguments** by sinking GPU dimension operations into the launch body before outlining. This means:
- For sgemm, cta, stencil3d: No synthetic args remain after sinking
- For dotproduct, sgemm2: No synthetic args after sinking

However, this infrastructure might still be useful for edge cases where:
- Values derived from GPU dims cannot be sunk (side-effects)
- Future kernels have unusual dimension computations

**Recommendation:** Keep for robustness, but most kernels won't use it.

---

### 4. blockDimXY Semantic Detection (`b67e4fa`)

**Purpose:** Detect `blockDim.x * blockDim.y` patterns for 2D shared memory indexing.

**Changes Made:**
- Added `isDim3MemrefType()` - detects `memref<?x3xi32>` dim3 types
- Added `findDim3ArgPositions()` - locates dim3 args dynamically
- Added blockDimXY, blockDimXZ, blockDimYZ pattern detection
- Updated `getGpuDimIndex()` to use dynamic positions

**Still Needed:** ⚠️ **PARTIALLY**

**Reason:** Similar to semantic inference, this was designed to handle synthetic arguments for 2D block dimension products. With `SinkGpuDimsIntoLaunch`:
- The `blockDim.x * blockDim.y` computation is sunk into the launch body
- It becomes part of the kernel code, not a synthetic argument

**Recommendation:** Keep for robustness, but less critical now.

---

### 5. Vararg Support (`26eb428`)

**Purpose:** Add vararg call support in LLVM dialect.

**Changes Made:**
- Added `callee_type` attribute for vararg calls
- Modified LLVM::CallOp and InvokeOp

**Still Needed:** ✅ **YES** (but for different reasons)

**Reason:** This is an upstream LLVM change that was merged into the Polygeist fork. It's needed for:
- `printf` calls which are variadic
- Any other variadic function calls from kernels

This is not Vortex-specific but is required for correct LLVM lowering.

---

### 6. Uncommitted Change: Remove launch_wrapper Skip

**Purpose:** Outline ALL `gpu.launch` ops, including those with `polygeist.launch_wrapper` attribute.

**Current State:**
```cpp
// OLD (being removed):
if (op->hasAttr("polygeist.launch_wrapper")) {
  return WalkResult::advance();  // Skip wrapper launches
}

// NEW:
// Note: We outline ALL gpu.launch ops...
```

**Still Needed:** ✅ **YES**

**Reason:** The skip logic was causing 2D/3D kernels to not be outlined properly. All launch ops need to be outlined for proper kernel extraction.

**Recommendation:** Commit this change.

---

## Summary Table

| Change | Still Needed | Reason |
|--------|--------------|--------|
| kernel_arg_mapping | ✅ YES | Core infrastructure for argument ordering |
| Vortex metadata | ✅ YES | Required for kernel naming and arg marshaling |
| Synthetic semantic inference | ⚠️ PARTIAL | Most cases eliminated by dimension sinking |
| blockDimXY detection | ⚠️ PARTIAL | Most cases eliminated by dimension sinking |
| Vararg support | ✅ YES | Required for printf and variadic functions |
| Remove launch_wrapper skip | ✅ YES | Required for proper kernel outlining |

---

## Recommendations

### Immediate Actions

1. **Commit the uncommitted change** - The launch_wrapper skip removal is critical and should be committed.

2. **Keep all current changes** - Even the "partial" ones provide robustness for edge cases.

### Future Considerations

1. **Consider upstreaming kernel_arg_mapping** - This is generally useful for any backend that needs host-device argument correspondence. Could be proposed to MLIR upstream.

2. **Document synthetic semantic handling** - Add comments explaining that this is a fallback for cases where dimension sinking doesn't eliminate all synthetic args.

3. **Monitor for upstream changes** - The vararg support change is from upstream; ensure Polygeist stays in sync.

### Cleanup Opportunities

1. **Remove unused semantic patterns** - If certain semantic patterns (like `blockDim/2`) are never hit due to dimension sinking, they could be removed to simplify the code.

2. **Simplify dim3 detection** - If dim3 positions are always consistent after source transformation, the dynamic detection could be simplified.

---

## Path to Eliminating llvm-project Modifications

**Long-term Goal:** Eliminate all modifications to the `llvm-project` submodule so that:
1. The Vortex integration can use upstream Polygeist/LLVM
2. Integration into the primary Vortex repo requires no forked dependencies
3. Maintenance burden is reduced

### Analysis: Can We Move Functionality to Polygeist Passes?

| Change | Can Move to Polygeist? | Strategy |
|--------|------------------------|----------|
| kernel_arg_mapping | ✅ YES | Post-outlining analysis pass |
| Vortex metadata | ✅ YES | Pre-outlining attribute injection |
| Synthetic semantic inference | ✅ YES | Already largely eliminated by SinkGpuDimsIntoLaunch |
| blockDimXY detection | ✅ YES | Already largely eliminated by SinkGpuDimsIntoLaunch |
| Vararg support | ❌ NO | Upstream change, already merged to LLVM main |
| launch_wrapper skip | ✅ YES | Can be handled in driver.cc pipeline ordering |

### Proposed Refactoring Steps

#### Step 1: Move kernel_arg_mapping to Polygeist Pass

Create a new `ComputeKernelArgMapping` pass in Polygeist that:
1. Runs **after** `gpu-kernel-outlining`
2. Analyzes `gpu.launch_func` to find the wrapper function
3. Traces `gpu.func` arguments back to wrapper function arguments
4. Sets `kernel_arg_mapping` attribute on `gpu.func`

**Complexity:** Medium
**Files to create:** `lib/polygeist/Passes/ComputeKernelArgMapping.cpp`

```cpp
// Pseudo-code for the pass
void runOnOperation() {
  getOperation().walk([&](gpu::GPUFuncOp gpuFunc) {
    // Find the launch_func that calls this kernel
    auto launchFunc = findLaunchFuncForKernel(gpuFunc);

    // Find the wrapper function containing the launch_func
    auto wrapper = launchFunc->getParentOfType<func::FuncOp>();

    // Trace kernel args to wrapper args
    SmallVector<int64_t> mapping = computeArgMapping(gpuFunc, wrapper);

    gpuFunc->setAttr("kernel_arg_mapping", ...);
  });
}
```

#### Step 2: Move Vortex Metadata Propagation to Polygeist

The `vortex.kernel_args`, `vortex.kernel_args_size`, `vortex.kernel_name` attributes are set by `HIPSourceTransform.cc` on the `gpu.launch` during AST-to-MLIR conversion.

**Option A:** Keep setting on `gpu.launch`, add a Polygeist pass to copy to `gpu.func` after outlining.

**Option B:** Set attributes directly on `gpu.func` if we control the outlining process.

**Complexity:** Low
**Files to modify:** Add attribute copying to `ComputeKernelArgMapping` pass

#### Step 3: Rely on SinkGpuDimsIntoLaunch for Synthetic Args

The `SinkGpuDimsIntoLaunch` pass already eliminates most synthetic arguments. With proper ordering in driver.cc:

```
1. SinkGpuDimsIntoLaunch (sinks dimension ops)
2. gpu-kernel-outlining (standard upstream)
3. ComputeKernelArgMapping (new Polygeist pass)
```

The synthetic semantic inference becomes unnecessary because:
- Dimension ops are inside the kernel body, not captured as arguments
- No synthetic args means no need to infer semantics

**Complexity:** Already done

#### Step 4: Handle launch_wrapper Skip in Pipeline

Instead of modifying KernelOutlining.cpp, ensure the pipeline never creates `gpu.launch` ops with `polygeist.launch_wrapper` that shouldn't be outlined.

**Approach:**
1. Review `ParallelLower` pass to understand wrapper attribute usage
2. Ensure attribute is only used for its intended purpose (preventing GPUWrapper conversion)
3. Remove the skip logic requirement by restructuring pass ordering

**Complexity:** Low-Medium

### Implementation Roadmap

| Phase | Task | Effort | Prerequisites |
|-------|------|--------|---------------|
| 1 | Create `ComputeKernelArgMapping` pass | 2-3 days | None |
| 2 | Add Vortex metadata copying to new pass | 1 day | Phase 1 |
| 3 | Remove synthetic semantic inference from KernelOutlining | 1 day | Phases 1-2 |
| 4 | Verify all tests pass with upstream KernelOutlining | 1 day | Phases 1-3 |
| 5 | Update driver.cc to use new pass ordering | 0.5 day | Phases 1-4 |
| 6 | Remove llvm-project modifications | 0.5 day | All phases |

**Total Estimated Effort:** 5-7 days

### Benefits of This Approach

1. **No llvm-project fork required** - Use upstream LLVM/MLIR directly
2. **Easier maintenance** - All Vortex-specific code in Polygeist
3. **Simpler integration** - Vortex repo can point to upstream Polygeist
4. **Better separation of concerns** - MLIR handles generic GPU, Polygeist handles Vortex

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Post-outlining analysis may miss information | Keep SinkGpuDimsIntoLaunch pre-outlining |
| Performance regression | Benchmark before/after |
| Edge cases not covered | Comprehensive test suite |

---

## Test Validation

After the `SinkGpuDimsIntoLaunch` pass, the following tests pass without needing synthetic argument handling:

| Test | Uses Synthetic Args Before | After Sinking |
|------|---------------------------|---------------|
| sgemm (2D) | Yes (gridDim, blockDim) | No |
| cta (2D) | Yes (gridDim, blockDim) | No |
| stencil3d (3D) | Yes (gridDim, blockDim) | No |
| dotproduct | Yes (blockDim for reduction) | No |
| sgemm2 | Yes (blockDim.x * blockDim.y) | No |

This validates that the dimension sinking approach successfully eliminates synthetic arguments at the source level, reducing reliance on the semantic inference infrastructure in KernelOutlining.cpp.

---

## Appendix: File Locations

| Component | Location |
|-----------|----------|
| KernelOutlining.cpp | `Polygeist/llvm-project/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp` |
| SinkGpuDimsIntoLaunch.cpp | `Polygeist/lib/polygeist/Passes/SinkGpuDimsIntoLaunch.cpp` |
| ReorderGPUKernelArgs.cpp | `Polygeist/lib/polygeist/Passes/ReorderGPUKernelArgs.cpp` |
| GenerateVortexMain.cpp | `Polygeist/lib/polygeist/Passes/GenerateVortexMain.cpp` |
| ConvertGPUToVortex.cpp | `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp` |
