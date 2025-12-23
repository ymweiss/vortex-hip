# Cgeist Pipeline Documentation

This document describes how `cgeist` (Clang to GPU/MLIR) processes HIP/CUDA source code through various stages to produce MLIR suitable for GPU execution.

## Overview

`cgeist` is a C/C++/CUDA/HIP to MLIR compiler that extends Clang's AST frontend with MLIR code generation. For GPU code, it handles both host and device code paths, generating appropriate MLIR dialects for each.

## Key Files

| File | Purpose |
|------|---------|
| `tools/cgeist/driver.cc` | Main driver, sets up compilation pipeline |
| `tools/cgeist/Lib/clang-mlir.cc` | AST to MLIR conversion (MLIRScanner) |
| `tools/cgeist/Lib/CGCall.cc` | Function call handling, GPU kernel launches |
| `tools/cgeist/Lib/CGStmt.cc` | Statement handling |
| `tools/cgeist/Lib/HIPSourceTransform.cc` | HIP source preprocessing (wrapper generation) |
| `tools/cgeist/Lib/HIPKernelAnalysis.cc` | Kernel argument extraction |

## Compilation Stages

### Stage 1: Source Preprocessing (HIPSourceTransform)

When `--transform-only` is used, cgeist performs AST-level transformations:

1. **Kernel Collection**: Traverses AST to find `__global__` functions
2. **Launch Site Detection**: Finds `hipLaunchKernelGGL` or `<<<>>>` calls
3. **Wrapper Generation**: Creates `__launch_<kernel>` wrapper functions
4. **Launch Replacement**: Replaces direct kernel launches with wrapper calls
5. **Stub Header Generation**: Creates `*_args.h` with argument metadata

```
Input HIP Source
       │
       ▼
┌──────────────────────────────┐
│  HIPSourceTransformer        │
│  - Find __global__ kernels   │
│  - Find launch sites         │
│  - Insert wrapper functions  │
│  - Replace launch calls      │
└──────────────────────────────┘
       │
       ▼
Transformed Source + Stub Headers
```

### Stage 2: AST to MLIR (MLIRScanner)

The `MLIRASTConsumer` in `clang-mlir.cc` converts Clang AST to MLIR:

1. **Function Declaration**: Creates `func.func` operations
2. **Statement Handling**: Converts control flow, loops, etc.
3. **Expression Handling**: Converts arithmetic, calls, etc.
4. **GPU Kernel Handling**: Special handling for `__global__` functions

For GPU code, kernel launches (`<<<>>>`) become `gpu.launch` operations:

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gx, %gy, %gz)
           threads(%tx, %ty, %tz) in (%blx, %bly, %blz) {
  // Kernel body
  gpu.terminator
}
```

### Stage 3: MLIR Optimization Passes

The driver sets up a pass pipeline based on flags. For GPU code (`-emit-cuda` or `-emit-rocm`):

```cpp
// Early GPU passes (before inliner)
if (EmitGPU) {
  pm.addPass(polygeist::createSinkIndexCastsIntoGPULaunchPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(polygeist::createMergeGPUModulesPass());
  pm.addPass(polygeist::createReorderGPUKernelArgsPass());
}

pm.addPass(mlir::createInlinerPass());
// ... more optimization passes ...

// Late GPU passes
if (EmitGPU) {
  pm.addPass(polygeist::createSinkIndexCastsIntoGPULaunchPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(polygeist::createMergeGPUModulesPass());
  pm.addPass(polygeist::createReorderGPUKernelArgsPass());
  pm.addPass(polygeist::createConvertParallelToGPUPass2());
  // ... GPU-specific passes ...
}
```

## Key Passes

### SinkIndexCastsIntoGPULaunch

**File**: `lib/polygeist/Passes/SinkIndexCastsIntoGPULaunch.cpp`

**Purpose**: Reduces kernel parameters by moving `arith.index_cast` operations into `gpu.launch` regions.

**Problem it solves**: When a value is used both as its original type AND as an index (e.g., for loop bounds), both values become kernel parameters. This pass sinks the cast inside the kernel.

**Before**:
```mlir
%casted = arith.index_cast %count : i32 to index
gpu.launch ... args(%count : i32, %casted : index) {
  scf.for %i = %c0 to %casted step %c1 { ... }
}
```

**After**:
```mlir
gpu.launch ... args(%count : i32) {
  %casted = arith.index_cast %count : i32 to index
  scf.for %i = %c0 to %casted step %c1 { ... }
}
```

### GpuKernelOutlining (MLIR built-in)

**Purpose**: Converts `gpu.launch` regions into `gpu.func` operations and `gpu.launch_func` calls.

**Before**:
```mlir
gpu.launch blocks(...) threads(...) {
  // Kernel body with captured values
  gpu.terminator
}
```

**After**:
```mlir
gpu.module @gpu_module {
  gpu.func @kernel(%arg0: ..., %arg1: ...) kernel {
    // Kernel body
    gpu.return
  }
}

gpu.launch_func @gpu_module::@kernel blocks(...) threads(...)
    args(%captured0 : ..., %captured1 : ...)
```

**Critical Issue**: The outlining pass collects captured values using `getUsedValuesDefinedAbove()`, which returns values in **first-use order**, not source order. This can reorder kernel arguments.

### ReorderGPUKernelArgs

**File**: `lib/polygeist/Passes/ReorderGPUKernelArgs.cpp`

**Purpose**: Fixes kernel argument order by matching against wrapper function parameters and sets `kernel_arg_mapping` attribute.

**How it works**:
1. Finds wrapper functions (e.g., `__launch_<kernel>`)
2. Extracts expected argument order from wrapper parameters
3. **Checks if GPU arg types differ from wrapper types** (fixed Dec 2025)
4. If types differ: Reorders `gpu.func` arguments to match wrapper order
5. Sets `kernel_arg_mapping` to identity `[0, 1, 2, ...]` after reordering
6. Sets `vortex.kernel_name` attribute for metadata file naming

**Key Fix (December 2025)**: The pass previously assumed Polygeist always reordered args to scalars-first, but this was inconsistent. The fix checks actual GPU arg types against wrapper types:

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

**Important**: This pass runs inside `cgeist` (not `polygeist-opt`). When modifying it, rebuild both `cgeist` and `polygeist-opt`.

**Limitation**: Only works if wrapper function exists and has correct parameter order.

### ConvertGPUToVortex

**File**: `lib/polygeist/Passes/ConvertGPUToVortex.cpp`

**Purpose**: Converts GPU dialect operations to Vortex-compatible form.

**Key transformations**:
- `gpu.thread_id` → `vx_get_threadIdx()` call
- `gpu.block_id` → `vx_get_blockIdx()` call
- `gpu.block_dim` → `vx_get_blockDim()` call
- `gpu.grid_dim` → `vx_get_gridDim()` call
- `gpu.barrier` → `__syncthreads()` call
- `gpu.shuffle` → Vortex shuffle intrinsics

### StripHostOnlyFunctions

**File**: `lib/polygeist/Passes/StripHostOnlyFunctions.cpp`

**Purpose**: Removes host-only functions (like `main()`) from device MLIR.

**How it works**: Functions with `polygeist.host_only_func` attribute are removed before device code generation.

## Argument Order Problem

### Root Cause

When `gpu.launch` is outlined to `gpu.func`, the captured values are collected in first-use order:

```cpp
// In MLIR's GpuKernelOutlining.cpp
llvm::SetVector<Value> operands;
getUsedValuesDefinedAbove(launchOp.body(), operands);
// operands now contains values in order of first use in kernel body
```

If the kernel uses arguments in a different order than declared:
```cpp
__global__ void kernel(float* a, float* b, int n) {
  if (threadIdx.x < n) {    // 'n' used first
    b[threadIdx.x] = a[threadIdx.x];  // 'a', 'b' used later
  }
}
```

The outlined function may have: `func @kernel(%n: i32, %a: ptr, %b: ptr)` instead of `(%a: ptr, %b: ptr, %n: i32)`.

### Tracing Mechanism

The `kernel_arg_mapping` attribute attempts to fix this:

```cpp
// In GpuKernelOutlining (modified by Polygeist)
for (Value operand : operands) {
  int argIndex = traceToFunctionArg(operand);
  argMapping.push_back(argIndex);  // -1 if tracing failed
}
```

The `traceToFunctionArg()` function walks the SSA def-use chain to find if the value comes from a function argument. If successful, it returns the argument index; if not, returns -1.

### When Tracing Fails

Tracing fails when:
1. Value comes from a local variable, not a function argument
2. Value passed through complex operations (casts, loads, etc.)
3. Wrapper function was inlined before outlining

When all mappings are -1, the pass falls back to **heuristic ordering** (pointers first, scalars last), which may not match the original order.

### Synthetic Arguments

When a loop bound is a kernel argument cast to index:
```cpp
for (int i = 0; i < count; ++i) { ... }
```

If `SinkIndexCastsIntoGPULaunch` doesn't run or doesn't catch this case, the outlined kernel gets an extra argument:
```mlir
gpu.func @kernel(%count: i32, %count_as_index: index, ...)
```

This creates a mismatch between host stub (5 args) and kernel (6 args).

## Debugging Tips

### View MLIR at Each Stage

Use `--keep-temps` with `compile_hip_v2.sh` to preserve intermediate files:
- `*_transformed.cu` - After HIPSourceTransform
- `*.gpu.mlir` - After cgeist, before Vortex lowering
- `*.vortex.mlir` - After ConvertGPUToVortex

### Check Kernel Arguments

Look for these attributes in GPU MLIR:
```mlir
gpu.func @main_kernel(%arg0: ..., %arg1: ...) kernel attributes {
  kernel_arg_mapping = array<i64: 0, 1, 2, -1>,  // -1 = tracing failed
  vortex.kernel_name = "my_kernel",
  vortex.num_synthetic_args = 1 : i32  // Extra args from casts
}
```

### Enable Pass Debug Output

Some passes have debug output:
```
[SinkIndexCasts] Sunk index_cast into gpu.launch
[ReorderGPUKernelArgs] Found wrapper function: __launch_kernel
[ReorderGPUKernelArgs] Reordering kernel args: [2, 1, 0, 3]
```

## Common Issues

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| Argument mismatch | Wrong values in kernel | Arg order different between host/device | Check `kernel_arg_mapping`, fix tracing |
| Synthetic arguments | Extra kernel arg | Loop bound cast not sunk | Run `SinkIndexCastsIntoGPULaunch` earlier |
| Empty wrapper | ReorderGPUKernelArgs finds nothing | Wrapper inlined/empty | Ensure wrapper has kernel launch |
| Scalar before pointer | Scalars first in outlined kernel | Tracing failed, heuristic used | Fix tracing or reorder in pass |

## Pass Order Sensitivity

The order of passes is critical:

```
1. SinkIndexCastsIntoGPULaunch  ← Must run before outlining
2. GpuKernelOutlining           ← Creates gpu.func from gpu.launch
3. MergeGPUModules              ← Combines GPU modules
4. ReorderGPUKernelArgs         ← Fixes arg order using wrappers
5. Inliner                      ← May inline wrappers (run AFTER reorder)
```

Running the inliner before reordering can destroy the wrapper functions needed for argument order recovery.
