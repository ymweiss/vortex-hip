# Kernel Argument Marshaling Fix

This document describes the changes made to fix HIP kernel argument marshaling for Vortex.

## Problem Summary

The vecadd kernel was compiling and loading on Vortex SimX, but producing wrong results due to argument marshaling mismatch between host and device.

**Root Cause:** Polygeist transforms kernel arguments during GPU lowering, adding computed values and reordering args. The host was passing HIP-style args but the kernel expected transformed args.

### Original HIP Kernel
```cpp
vecadd_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points)
```

### Polygeist-Transformed Kernel Arguments
After GPU lowering, Polygeist generates a kernel with 6 arguments:
```
arg0: index       (block_dim.x as index type)
arg1: i32         (block_dim.x as i32)
arg2: i32         (num_points)
arg3: memref      (src0)
arg4: memref      (src1)
arg5: memref      (dst)
```

The first two arguments are derived from `block_dim.x`, not passed by the user.

## Changes Made

### 1. GenerateVortexMain.cpp - Derive arg0/arg1 from block_dim

**File:** `Polygeist/lib/polygeist/Passes/GenerateVortexMain.cpp`

The `kernel_body` wrapper function now reads the first two kernel arguments from the `block_dim[0]` field in the header instead of user arguments:

```cpp
// Pre-load block_dim[0] for the first two args (if needed)
Value blockDimX_i32 = nullptr;
Value blockDimX_i64 = nullptr;
if (numLeadingScalars >= 2) {
  SmallVector<LLVM::GEPArg> gepIndices;
  gepIndices.push_back(static_cast<int32_t>(BLOCK_DIM_OFFSET));  // offset 12
  auto blockDimPtr = builder.create<LLVM::GEPOp>(...);
  blockDimX_i32 = builder.create<LLVM::LoadOp>(loc, i32Type, blockDimPtr);
  blockDimX_i64 = builder.create<LLVM::ZExtOp>(loc, i64Type, blockDimX_i32);
}
```

### 2. ConvertGPUToVortex.cpp - Skip block_dim args in metadata

**File:** `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp`

The metadata generation now skips the first 2 leading scalar arguments since they're derived from the block_dim header:

```cpp
// Skip first 2 leading scalars (derived from block_dim[0])
unsigned argsToSkip = (numLeadingScalars >= 2) ? 2 : 0;

for (auto argType : argTypes) {
  if (argIndex < argsToSkip) {
    argIndex++;
    continue;  // Skip - comes from block_dim header
  }
  // ... generate metadata for remaining args
}
```

### 3. hip_vortex_runtime.h - Add VortexKernelArgMeta type

**File:** `runtime/hip_vortex_runtime/include/hip_vortex_runtime.h`

Added type definitions for kernel argument metadata:

```cpp
typedef struct VortexKernelArgMeta {
    uint32_t offset;      // Offset in the args buffer
    uint32_t size;        // Size in bytes
    uint32_t is_pointer;  // 1 if device pointer, 0 for scalar
} VortexKernelArgMeta;

typedef struct VortexKernelArgs {
    uint32_t grid_dim[3];   // 12 bytes
    uint32_t block_dim[3];  // 12 bytes
    // User arguments follow
} VortexKernelArgs;
```

Also added `vortexLaunchKernel` function declaration.

### 4. vortexLaunchKernel - Handle host/device pointer size mismatch

**File:** `runtime/hip_vortex_runtime/src/hip_kernel.cpp`

The key fix was handling the difference between 64-bit host pointers and 32-bit device pointers:

```cpp
// Track separate offsets for host (8-byte ptrs) vs device (4-byte ptrs)
size_t device_offset = 0;
for (size_t i = 0; i < num_args; i++) {
    const VortexKernelArgMeta& meta = metadata[i];

    if (meta.is_pointer) {
        // Read 8-byte host pointer from meta.offset
        void* host_handle;
        memcpy(&host_handle, src_args + meta.offset, sizeof(void*));

        // Convert to 4-byte device address
        uint32_t device_addr = ...;

        // Write 4-byte device addr at device_offset
        memcpy(dst_args + device_offset, &device_addr, sizeof(uint32_t));
        device_offset += 4;
    } else {
        // Scalar: read from host, write to device
        memcpy(dst_args + device_offset, src_args + meta.offset, meta.size);
        device_offset += meta.size;
    }
}
```

### 5. generate_host_stubs.py - Compute host struct offsets

**File:** `scripts/generate_host_stubs.py`

Updated to compute host struct offsets accounting for 8-byte void* on 64-bit systems:

```python
# Compute host struct offsets (8-byte void* on 64-bit)
host_offset = 0
for arg in args:
    host_size = 8 if arg.get("is_pointer", False) else arg["size"]
    # Emit metadata with host_offset
    host_offset += host_size
```

### 6. inject_kernel_launchers.py - Include kernel_stubs.h

**File:** `scripts/polygeist/inject_kernel_launchers.py`

Updated to include `kernel_stubs.h` after `hip/hip_runtime.h` in the transformed source:

```python
remaining_with_stubs = re.sub(
    r'(#include\s*<hip/hip_runtime\.h>)',
    r'\1\n#include "kernel_stubs.h"',
    remaining, count=1
)
```

## Memory Layout

### Device Args Buffer (what kernel expects)
```
Offset   Size   Field
------   ----   -----
0        12     grid_dim[3]
12       12     block_dim[3]     <- arg0/arg1 derived from block_dim[0]
24       4      num_points       <- arg2 (user arg)
28       4      src0 dev addr    <- arg3 (4-byte RV32 pointer)
32       4      src1 dev addr    <- arg4
36       4      dst dev addr     <- arg5
------
Total: 40 bytes
```

### Host Args Struct (what launcher passes)
```
Offset   Size   Field
------   ----   -----
0        4      num_points (int32_t)
4        8      src0 (void* - 64-bit host pointer)
12       8      src1 (void*)
20       8      dst (void*)
------
Total: 28 bytes
```

The `vortexLaunchKernel` function reads from host offsets and writes to device offsets, handling the size difference.

## Testing

After these fixes, the vecadd test passes:
```
$ VORTEX_DRIVER=simx ./build_vecadd_test/vecadd
number of points: 16
...
PASSED!
```

## Additional Fixes (December 2025)

### Problem: Kernel Naming and Argument Order Mismatch

After kernel outlining, the gpu.func was named `main_kernel` (from the parent function) instead of the original `vecadd_kernel`. The ReorderGPUKernelArgsPass couldn't match the kernel to the wrapper, and the host stub expected a different kernel name than the compiled binary.

### Solution: Multi-Level Fixes

#### 1. ReorderGPUKernelArgs - Match by Arg Count
**File:** `Polygeist/lib/polygeist/Passes/ReorderGPUKernelArgs.cpp`

When exact name matching fails, try matching by argument count:
```cpp
// 3. If still no match, try matching by arg count
if (it == kernelArgIsPointer.end()) {
  for (auto &entry : kernelArgIsPointer) {
    if (entry.second.size() == numGpuUserArgs) {
      it = kernelArgIsPointer.find(entry.first());
      break;
    }
  }
}
```

Also set `vortex.kernel_name` attribute to propagate original kernel name:
```cpp
gpuFunc->setAttr("vortex.kernel_name",
                StringAttr::get(ctx, originalKernelName));
```

#### 2. ConvertGPUToVortex - Use vortex.kernel_name
**File:** `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp`

Use the `vortex.kernel_name` attribute for metadata file naming:
```cpp
if (auto kernelNameAttr = funcOp->getAttrOfType<StringAttr>("vortex.kernel_name")) {
  meta.kernelName = kernelNameAttr.getValue().str();
} else {
  meta.kernelName = extractBaseKernelName(funcOp.getName()).str();
}
```

#### 3. HIPSourceTransform - Conditional Wrapper
**File:** `Polygeist/tools/cgeist/Lib/HIPSourceTransform.cc`

Generate conditional wrapper that calls stub on host:
```cpp
os << "#ifdef HIP_HOST_COMPILATION\n";
os << "void " << wrapperName << "(" << params << ") {\n";
os << "    " << stubName << "(__grid, __block, " << args << ");\n";
os << "}\n";
os << "#else\n";
os << "void " << wrapperName << "(" << params << ") {\n";
os << "    " << kernel.demangledName << "<<<__grid, __block>>>(" << args << ");\n";
os << "}\n";
os << "#endif\n";
```

#### 4. compile_hip_v2.sh - Use Transformed Source
**File:** `scripts/compile_hip_v2.sh`

Use the transformed source for host compilation:
```bash
HOST_SOURCE="$WORK_DIR/${BASENAME}_transformed.cu"
cp "$TRANSFORMED_CU" "$HOST_SOURCE"
# ...
COMPILE_SOURCE="${HOST_SOURCE:-$INPUT_FILE}"
```

### Result

The end-to-end pipeline now correctly:
1. Preserves kernel argument order (ptr, ptr, ptr, scalar)
2. Names kernel binary to match host stub expectations
3. Marshals host pointers to device addresses via metadata

## Files Modified

| File | Change |
|------|--------|
| `Polygeist/lib/polygeist/Passes/GenerateVortexMain.cpp` | Derive arg0/arg1 from block_dim[0] |
| `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp` | Skip leading block_dim args; use vortex.kernel_name |
| `Polygeist/lib/polygeist/Passes/ReorderGPUKernelArgs.cpp` | Match by arg count; set vortex.kernel_name |
| `Polygeist/tools/cgeist/Lib/HIPSourceTransform.cc` | Conditional wrapper generation |
| `runtime/hip_vortex_runtime/include/hip_vortex_runtime.h` | Add VortexKernelArgMeta, vortexLaunchKernel |
| `runtime/hip_vortex_runtime/src/hip_kernel.cpp` | Handle host/device offset mapping |
| `scripts/compile_hip_v2.sh` | Use transformed source for host |
| `scripts/generate_host_stubs.py` | Compute host struct offsets for 64-bit |
| `scripts/polygeist/inject_kernel_launchers.py` | Include kernel_stubs.h |

## Latest Fix: kernel_arg_mapping Computation (December 22, 2025)

### Problem: Incorrect Mapping for Non-Reordered Kernels

The `ReorderGPUKernelArgs` pass was computing incorrect `kernel_arg_mapping` values when Polygeist didn't reorder kernel arguments to scalars-first order.

**Example - vecadd kernel:**
- Wrapper order: `(ptr, ptr, ptr, scalar)` - original HIP order
- GPU kernel order: `(memref, memref, memref, i32)` - NOT reordered by Polygeist
- Computed mapping: `[3, 0, 1, 2]` - WRONG (assumed scalar-first)
- Correct mapping: `[0, 1, 2, 3]` - identity

### Root Cause

The pass assumed Polygeist ALWAYS reorders to scalars-first, but Polygeist's reordering behavior is inconsistent across kernels.

### Solution

Modified `ReorderGPUKernelArgs.cpp` to:

1. **Check actual GPU arg types** vs wrapper types:
   ```cpp
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

2. **Conditionally reorder**: Only reorder if types don't match wrapper order.

3. **Set identity mapping after reorder**: After reordering kernel args to match wrapper, set `kernel_arg_mapping = [0, 1, 2, ...]`.

4. **Handle leading synthetic args**: Count leading `index` types separately from trailing `llvm.ptr` types.

### Key Insight: cgeist Internal Pipeline

The `ReorderGPUKernelArgs` pass runs inside `cgeist` (not `polygeist-opt`) as part of its internal pass pipeline at:
- `tools/cgeist/driver.cc:776`
- `tools/cgeist/driver.cc:934`

**When modifying this pass, `cgeist` must be rebuilt**, not just `polygeist-opt`.

### Test Results

After the fix:
- vecadd: PASSED ✓
- mstress: PASSED ✓
- relu: PASSED ✓
- basic: PASSED ✓
- demo: PASSED ✓
