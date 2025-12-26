# Launch Wrapper Generation Investigation

## Status: IN PROGRESS

## Problem Statement

When `hipLaunchKernelGGL` is called directly in `main()`, kernel arguments are local variables, not function parameters. During kernel outlining in `KernelOutlining.cpp`, `traceToFunctionArg()` fails for all operands, resulting in `kernel_arg_mapping = [-1, -1, -1, -1]`. The sorting algorithm then reorders arguments incorrectly:

- **HIP Source Order:** `(src0*, src1*, dst*, num_points)` - pointers first, scalar last
- **Device Kernel Order:** `(num_points, src0*, src1*, dst*)` - scalar first, pointers last

## Solution Approach

Generate launch wrapper functions during clang AST parsing where kernel arguments ARE function parameters, ensuring `traceToFunctionArg()` succeeds and argument order is preserved.

**Critical Constraint:** No changes can be made to LLVM/MLIR core code (`Polygeist/llvm-project/`) because reducing storage requirements requires using builtin LLVM instead of a modified fork.

## Investigation Findings

### 1. Wrapper Linkage Change (COMPLETED)

**File:** `Polygeist/tools/cgeist/Lib/clang-mlir.cc:5382-5389`

Changed wrapper linkage from Internal to External so wrappers can be exported as object files:

```cpp
// Before:
SymbolTable::setSymbolVisibility(wrapperFunc, SymbolTable::Visibility::Private);
attrs.set("llvm.linkage", mlir::LLVM::LinkageAttr::get(builder.getContext(),
                                                        mlir::LLVM::Linkage::Internal));

// After:
SymbolTable::setSymbolVisibility(wrapperFunc, SymbolTable::Visibility::Public);
attrs.set("llvm.linkage", mlir::LLVM::LinkageAttr::get(builder.getContext(),
                                                        mlir::LLVM::Linkage::External));
```

### 2. Auto-Generated Wrappers Have Empty Bodies

When using `--emit-vortex-wrappers` flag, the generated wrapper has correct signature and attributes but **empty body**:

```mlir
func.func @__polygeist_launch__Z12basic_kernelPiS_j(
    %arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32,
    %arg3: index, %arg4: index, %arg5: index,
    %arg6: index, %arg7: index, %arg8: index)
    attributes {llvm.linkage = #llvm.linkage<external>,
                vortex.kernel_args = [#vortex.kernel_arg<...>]} {
  return  // <-- EMPTY BODY - no gpu.launch operation
}
```

The wrapper generation code at `clang-mlir.cc:5413-5434` SHOULD create a `gpu.launch` operation with the kernel call inside, but the body is not being populated.

### 3. Working Case vs Non-Working Case

**Working Case (basic.hip with hipLaunchKernelGGL in main()):**
```mlir
// gpu.launch_func is properly generated
gpu.launch_func @__polygeist_gpu_module::@main_kernel
    blocks in (%56, %c1, %c1) threads in (%c16, %c1, %c1)
    args(%3#0 : i32, %54 : memref<?xi32>, %55 : memref<?xi32>)
```

**Non-Working Case (vecadd.hip with user-defined wrapper):**
```mlir
// Empty async.execute block instead of gpu.launch_func
async.execute {
  async.yield
}
```

### 4. Root Cause Analysis

**Discovery:** When `hipLaunchKernelGGL` is called inside a **non-main function**, cgeist doesn't have access to the kernel function definition. In host compilation mode, device functions (`__global__` kernels) aren't compiled, so the kernel call is effectively dropped.

**Key insight:** The `--emit-vortex-wrappers` flag operates in DEVICE mode (see `clang-mlir.cc:5720-5757`), but the wrapper body generation may not be properly inserting the `gpu.launch` operation.

The wrapper generation at `clang-mlir.cc:5413-5434` is designed to:
1. Create the wrapper function with correct signature
2. Create `gpu.launch` operation inside the wrapper
3. Call the kernel inside the launch block

However, the intermediate MLIR output shows the wrapper body is just `return` with no launch operation.

### 5. Relevant Code Locations

| File | Lines | Purpose |
|------|-------|---------|
| `clang-mlir.cc` | 5382-5389 | Wrapper linkage settings (MODIFIED) |
| `clang-mlir.cc` | 5413-5434 | Wrapper body generation with gpu.launch |
| `clang-mlir.cc` | 5720-5757 | EmitVortexWrappers mode implementation |
| `ConvertGPUToVortex.cpp` | ~400-500 | LaunchFuncMetadataExtraction pattern |

### 6. What Was Tried

1. **Changed wrapper linkage to External** - Successfully modified
2. **Tested with `--emit-vortex-wrappers --host-library`** - Wrappers generated but empty body
3. **Examined intermediate MLIR with `--output-intermediate-gpu=1`** - Confirmed empty bodies
4. **Compared working (basic.hip) vs non-working (vecadd.hip)** - Identified the main() vs wrapper function difference
5. **User-defined wrapper in vecadd.hip** - Produces empty async.execute block

## Next Steps for Investigation

1. **Debug wrapper body generation in clang-mlir.cc**
   - Add debug output to `getOrCreateLaunchWrapper()` around lines 5413-5434
   - Verify that the kernel function is found and `gpu.launch` is being created

2. **Check EmitVortexWrappers mode flow**
   - Trace the code path when `--emit-vortex-wrappers` is passed
   - Verify kernel function lookup in device compilation mode

3. **Investigate async.execute generation**
   - Understand why user-defined wrappers produce `async.execute` instead of `gpu.launch_func`
   - May need to trace through `HandleHIPLaunchKernelGGL()` function

4. **Alternative approach: Source-level wrappers**
   - If AST-level wrapper generation is too complex, consider requiring users to write launch wrappers in their HIP source
   - Document pattern for manual wrapper creation

## Related Documentation

- Plan file: `/home/yaakov/.claude/plans/frolicking-wondering-tower.md`
- MAINLINE_INTEGRATION_PLAN.md - needs LLVM constraint documentation added

## Test Commands

```bash
# Test wrapper generation
./Polygeist/build/bin/cgeist /tmp/basic.cu \
    -x hip --emit-vortex-wrappers \
    --output-intermediate-gpu=1 -o /tmp/basic_wrappers.gpu.mlir

# Check generated wrapper
grep -A 20 "__polygeist_launch" /tmp/basic_wrappers.gpu.mlir

# Compare with normal compilation
./Polygeist/build/bin/cgeist /tmp/basic.cu \
    -x hip --output-intermediate-gpu=1 -o /tmp/basic_normal.gpu.mlir
```
