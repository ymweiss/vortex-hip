# Analysis: Host-Device Argument Ordering for HIP-to-Vortex Pipeline

## Current State

The HIP-to-Vortex compilation pipeline currently has inconsistent argument ordering between host and device, causing kernel execution failures.

## The Problem

### Argument Flow in Current Pipeline

```
HIP Source → Polygeist (GPU lowering) → Vortex MLIR → LLVM IR → RISC-V Binary
                  ↓
           Host args buffer     Device kernel args
           (original order)     (Polygeist-reordered)
```

**Host side** (HIP runtime `vortexLaunchKernel`):
- Packs kernel arguments in **original HIP source order**
- Example for vecadd: `[a_ptr, b_ptr, c_ptr, size]` → offsets 24, 28, 32, 36

**Device side** (Polygeist-generated kernel):
- Reorders arguments by type: **scalars first, then pointers**
- Example for vecadd (expected): `(size, a_ptr, b_ptr, c_ptr)`
- But actual gpu.func: `(memref<?xf32>, memref<?xf32>, memref<?xf32>, i32)` - NOT reordered!

### Observed Behavior

For the vecadd kernel:
- **Wrapper original order**: `(a:ptr, b:ptr, c:ptr, size:i32)` → isPointer: `[true, true, true, false]`
- **Expected Polygeist reorder**: `(size:i32, a:ptr, b:ptr, c:ptr)` (scalars first)
- **Actual GPU kernel**: `(memref, memref, memref, i32)` (memrefs first, scalar last)

This mismatch means the `kernel_arg_mapping` computed in `ReorderGPUKernelArgs` doesn't match reality.

### kernel_arg_mapping Issue

The mapping `[3, 0, 1, 2]` was computed assuming:
- deviceArg 0 → hostArg 3 (size) - assumes scalar at front
- deviceArg 1 → hostArg 0 (a)
- deviceArg 2 → hostArg 1 (b)
- deviceArg 3 → hostArg 2 (c)

But the actual kernel has `(memref, memref, memref, i32)`:
- deviceArg 0 = memref (should be a, hostIdx 0)
- deviceArg 1 = memref (should be b, hostIdx 1)
- deviceArg 2 = memref (should be c, hostIdx 2)
- deviceArg 3 = i32 (should be size, hostIdx 3)

The correct mapping should be `[0, 1, 2, 3]` (identity).

## Root Cause Analysis

### Issue 1: Polygeist Reordering Not Happening

The `ReorderGPUKernelArgs` pass was designed to handle kernels that Polygeist had already reordered. But evidence shows the GPU kernel is NOT being reordered by Polygeist's GPU lowering passes.

### Issue 2: Inconsistent Assumptions

- `ReorderGPUKernelArgs` assumes Polygeist reorders to `(scalars, pointers)`
- `GenerateVortexMain` uses `kernel_arg_mapping` to load args from buffer
- HIP runtime packs args in original order

The mismatch creates three different orderings that don't align.

### Issue 3: mstress vs vecadd Difference

**mstress** (works after fixes):
- GPU kernel has extra synthetic args (total_threads for bounds check)
- The synthetic arg detection correctly identifies these
- After mapping fix, args load correctly

**vecadd** (broken):
- GPU kernel has NO synthetic args
- But mapping assumes scalar-first ordering that doesn't exist
- Results in loading wrong data

## Solution Options

### Option A: Fix ReorderGPUKernelArgs to Actually Reorder

Make the pass do what its name implies: reorder GPU kernel arguments to match the expected order.

**Pros:**
- Self-contained fix in one pass
- Kernel binaries will have consistent arg order

**Cons:**
- Complex implementation (need to update kernel body, call sites)
- May interfere with other Polygeist passes

### Option B: Match Host to Device Order

Modify the host side to pack arguments in Polygeist's expected order (scalars first, pointers second).

**Pros:**
- No kernel modifications needed
- Simpler conceptually

**Cons:**
- Requires changes to HIP runtime marshaling
- Need to know reorder mapping at runtime

### Option C: Device-Side Arg Unpacking with Correct Mapping (Current Approach)

Keep args in original order on host, compute correct mapping based on actual kernel signature.

**Pros:**
- Minimal changes needed
- Works with current pipeline

**Cons:**
- Complex mapping computation
- Fragile if Polygeist behavior changes

## Recommended Solution: Option B with Pipeline Support

### Architecture

```
HIP Source
    ↓
Polygeist (GPU lowering)
    ↓
ReorderGPUKernelArgs
    ├── 1. Detect wrapper arg types from source
    ├── 2. Compute reorder permutation (scalars first)
    ├── 3. Reorder GPU kernel args to match
    ├── 4. Update kernel_arg_mapping to [0,1,2,...] (identity)
    └── 5. Generate host stub with reordered arg packing
            ↓
Host Compilation (uses reordered stub)
    ├── Args packed in DEVICE order
    └── kernel_arg_mapping = identity
            ↓
GenerateVortexMain
    ├── Loads args sequentially (no remapping needed)
    └── Device kernel receives args directly
```

### Implementation Steps

1. **ReorderGPUKernelArgs Enhancement**
   - Actually reorder the `gpu.func` arguments
   - Update all references within kernel body
   - Update `gpu.launch_func` operand order
   - Set `kernel_arg_mapping` to identity `[0, 1, 2, ...]`

2. **Host Stub Generation**
   - Generate wrapper that packs args in reordered order
   - Or generate metadata for HIP runtime to do the reordering

3. **GenerateVortexMain Simplification**
   - With identity mapping, just load args sequentially
   - Remove complex offset computation based on hostIdx

### Key Insight

The fundamental issue is that we're trying to bridge two different conventions:
- **HIP**: Args in user-specified order
- **Polygeist/MLIR**: Args reordered by type for optimization

A clean solution reorders ONCE at the boundary (either host-side packing or device-side unpacking), not both.

## Files Involved

| File | Current Role | Needed Changes |
|------|--------------|----------------|
| `ReorderGPUKernelArgs.cpp` | Computes mapping, doesn't reorder | Actually reorder args |
| `GenerateVortexMain.cpp` | Complex offset mapping | Simplify to sequential load |
| `vortex_hip_runtime.cpp` | Packs args in original order | Pack in reordered order OR use metadata |
| Host stub generator | Generates identity wrapper | Generate reordering wrapper |

## Current Workaround Limitations

The current fix (mapping-based offset computation) works only when:
1. Polygeist doesn't reorder kernel args (vecadd case fails)
2. Polygeist does reorder but we detect synthetic args correctly (mstress case works)

This is fragile because it depends on Polygeist's internal behavior which may vary based on:
- Kernel complexity
- Number/types of arguments
- MLIR optimization level
- Polygeist version

## Investigation Findings (December 2025)

### Key Discovery: gpu.launch_func and gpu.func Are Already Consistent

Examination of the GPU MLIR reveals that **the `gpu.launch_func` operand order always matches the `gpu.func` signature order**:

**vecadd (no synthetic args):**
```mlir
gpu.func @main_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
gpu.launch_func @main_kernel args(%76 : memref<?xf32>, %77 : memref<?xf32>, %78 : memref<?xf32>, %3#0 : i32)
```
Order: (memrefs, i32) - NOT reordered

**mstress (has synthetic args):**
```mlir
gpu.func @main_kernel(%arg0: i32, %arg1: index, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>)
gpu.launch_func @main_kernel args(%5 : i32, %108 : index, %105 : memref<?xi32>, %106 : memref<?xf32>, %107 : memref<?xf32>)
```
Order: (i32, index, memrefs) - scalars first

### Why Current Implementation Fails

The `ReorderGPUKernelArgs` pass computes `kernel_arg_mapping` assuming Polygeist ALWAYS reorders to scalars-first. But:

1. Polygeist's reordering is inconsistent across kernels
2. The mapping computation uses `computeArgPermutation` which assumes scalar-first order
3. For vecadd: mapping = [3, 0, 1, 2] (assumes scalar first)
4. But actual kernel has memrefs first → mapping is WRONG

### The Solution: Host Library Compilation Path

The `ConvertGPULaunchToHostCall` pass already handles this correctly:

```cpp
// From ConvertGPULaunchToHostCall.cpp lines 466-494
unsigned argOffset = 24;
for (auto origArg : info.operands) {  // Uses gpu.launch_func operand order
  // ... convert and store ...
  storeI32AtOffset(argVal, argOffset);
  argOffset += 4;
}
```

This pass:
1. Takes operands from `gpu.launch_func` in their existing order
2. Stores them sequentially in the args buffer
3. Since `gpu.launch_func` order matches `gpu.func` order, device gets correct args!

### Path Forward: Use Host Library Compilation

Instead of trying to fix the mapping computation (which is complex and fragile), we should:

1. **Use the existing `ConvertGPULaunchToHostCall` pass** for host compilation
2. **Remove the complex mapping logic** from `GenerateVortexMain`
3. **Simplify `GenerateVortexMain`** to load args sequentially (no remapping)

This approach:
- Uses proven MLIR infrastructure
- Eliminates host-device ordering mismatches at the source
- Requires minimal new code (the pass already exists)

## Implementation Plan for Option B

### Phase 1: Enable Host Library Compilation Path

1. **Modify compile script** to use `--host-library` mode by default
2. **Add `vortex_launch_with_args` runtime function** if not present
3. **Test with existing kernels** (vecadd, mstress)

### Phase 2: Simplify Device-Side Arg Loading

1. **Remove `kernel_arg_mapping` computation** from `ReorderGPUKernelArgs`
2. **Simplify `GenerateVortexMain`** to load args sequentially
3. **Remove `computeArgPermutation` logic** (no longer needed)

### Phase 3: Cleanup

1. Remove unused mapping code
2. Update documentation
3. Test full kernel suite

## Implemented Solution (December 22, 2025)

### Final Fix: Check Actual GPU Arg Types

Instead of the host library compilation approach (which had issues with complex C++ types), the fix was implemented directly in `ReorderGPUKernelArgs.cpp`:

**Key Changes:**

1. **Check actual GPU arg types vs wrapper types** instead of assuming scalars-first order:
   ```cpp
   bool needsReorder = false;
   for (unsigned i = 0; i < numUserArgs; ++i) {
     Type gpuArgType = argTypes[argsToSkip + i];
     bool gpuIsPtr = gpuArgType.isa<MemRefType>();
     bool wrapperIsPtr = originalIsPointer[i];
     if (gpuIsPtr != wrapperIsPtr) {
       needsReorder = true;
       break;
     }
   }
   ```

2. **Only reorder when actually needed**: If GPU arg types already match wrapper order, skip reordering and set identity mapping.

3. **Set identity mapping after reordering**: When reordering happens, set `kernel_arg_mapping = [0, 1, 2, ...]` on the cloned kernel.

4. **Handle leading synthetic args separately**: Count leading `index` types as synthetic (e.g., total thread counts) distinct from trailing `llvm.ptr` types (captured globals).

### Why This Works

- If Polygeist reorders args (scalar-first): The pass reorders back to wrapper order, sets identity mapping
- If Polygeist doesn't reorder: No reordering needed, identity mapping set directly
- Either way: `GenerateVortexMain` loads args sequentially with identity mapping

### Test Results

All tests pass after the fix:
- vecadd: PASSED (was broken with [3,0,1,2] mapping)
- mstress: PASSED (still works)
- relu: PASSED
- basic: PASSED
- demo: PASSED

### Files Modified

| File | Change |
|------|--------|
| `ReorderGPUKernelArgs.cpp` | Check actual types, set identity mapping after reorder |

### Important Note: cgeist vs polygeist-opt

The `ReorderGPUKernelArgs` pass runs inside `cgeist` as part of its internal pass pipeline (lines 776 and 934 in `driver.cc`), NOT in `polygeist-opt`. When making changes to this pass, **both `cgeist` and `polygeist-opt` must be rebuilt**.

## Conclusion

The fix eliminates the assumption that Polygeist always reorders to scalars-first. By checking actual types and conditionally reordering, the pass now handles both cases correctly. This is simpler and more robust than the host library compilation approach.
