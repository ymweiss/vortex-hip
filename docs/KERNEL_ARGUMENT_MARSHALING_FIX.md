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

## Files Modified

| File | Change |
|------|--------|
| `Polygeist/lib/polygeist/Passes/GenerateVortexMain.cpp` | Derive arg0/arg1 from block_dim[0] |
| `Polygeist/lib/polygeist/Passes/ConvertGPUToVortex.cpp` | Skip leading block_dim args in metadata |
| `runtime/hip_vortex_runtime/include/hip_vortex_runtime.h` | Add VortexKernelArgMeta, vortexLaunchKernel |
| `runtime/hip_vortex_runtime/src/hip_kernel.cpp` | Handle host/device offset mapping |
| `scripts/generate_host_stubs.py` | Compute host struct offsets for 64-bit |
| `scripts/polygeist/inject_kernel_launchers.py` | Include kernel_stubs.h |
