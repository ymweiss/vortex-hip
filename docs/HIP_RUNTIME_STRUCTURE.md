# HIP Runtime Structure

This document explains the organization of the HIP runtime code in the vortex_hip repository.

## Overview

The repository contains **two** HIP runtime implementations in different locations. This is a historical artifact that should be consolidated in the future.

```
vortex_hip/
├── runtime/
│   ├── include/                    # "New" header location
│   │   ├── vortex_hip_runtime.h    # Main runtime header (PREFERRED)
│   │   ├── vortex_hip_device.h     # Device-side definitions
│   │   └── hip/
│   │       ├── hip_runtime.h       # HIP compatibility header
│   │       └── hip_runtime_polygeist.h
│   ├── src/
│   │   └── vortex_hip_runtime.cpp  # Older implementation (unused?)
│   ├── hip_vortex_runtime/         # "Active" implementation
│   │   ├── include/
│   │   │   ├── hip_vortex_runtime.h  # Alternative header
│   │   │   └── hip/
│   │   │       └── hip_runtime.h     # Points to ../hip_vortex_runtime.h
│   │   ├── src/
│   │   │   ├── hip_device.cpp      # Device management
│   │   │   ├── hip_memory.cpp      # Memory operations
│   │   │   ├── hip_kernel.cpp      # Kernel launch (vortexLaunchKernel)
│   │   │   └── hip_error.cpp       # Error handling
│   │   └── test/
│   │       └── ...
│   ├── examples/
│   └── build/                      # Build output (libhip_vortex.so)
└── hip_runtime_vortex/             # Device compilation headers
    └── hip_runtime.h               # For Polygeist device compilation
```

## Which Files Are Actually Used?

### For Host Compilation (the executable)

The **active** runtime is in `runtime/hip_vortex_runtime/`:

| File | Purpose |
|------|---------|
| `hip_vortex_runtime/include/hip_vortex_runtime.h` | Main HIP API header |
| `hip_vortex_runtime/src/hip_device.cpp` | `hipInit`, `hipSetDevice`, `hipDeviceSynchronize` |
| `hip_vortex_runtime/src/hip_memory.cpp` | `hipMalloc`, `hipFree`, `hipMemcpy` |
| `hip_vortex_runtime/src/hip_kernel.cpp` | `hipModuleLaunchKernel`, `vortexLaunchKernel` |
| `hip_vortex_runtime/src/hip_error.cpp` | `hipGetErrorString`, error handling |

These are compiled into `libhip_vortex.so`.

### For Device Compilation (Polygeist)

| File | Purpose |
|------|---------|
| `hip_runtime_vortex/hip_runtime.h` | Provides `__global__`, `threadIdx`, `blockIdx` for CUDA mode |

This header is used when Polygeist compiles device code with `-D__CUDA__`.

## Header Include Paths

### Host Code
```cpp
#include <hip/hip_runtime.h>  // Resolves to hip_vortex_runtime.h
```

Compile with:
```bash
g++ -I runtime/hip_vortex_runtime/include ...
```

### Device Code (Polygeist)
```cpp
#include "hip_runtime_vortex/hip_runtime.h"
```

The `inject_kernel_launchers.py` script transforms includes appropriately.

## Key Types and Functions

### VortexKernelArgs (header layout)
```cpp
typedef struct VortexKernelArgs {
    uint32_t grid_dim[3];   // Offset 0, 12 bytes
    uint32_t block_dim[3];  // Offset 12, 12 bytes
    // User arguments follow at offset 24
} VortexKernelArgs;
```

### VortexKernelArgMeta (argument descriptor)
```cpp
typedef struct VortexKernelArgMeta {
    uint32_t offset;      // Offset in host args struct
    uint32_t size;        // Size in bytes
    uint32_t is_pointer;  // 1 for device pointers
} VortexKernelArgMeta;
```

### vortexLaunchKernel (kernel launch)
```cpp
hipError_t vortexLaunchKernel(
    const char* kernel_name,
    dim3 gridDim,
    dim3 blockDim,
    const void* args,           // Pointer to packed args struct
    size_t args_size,
    const VortexKernelArgMeta* metadata,
    size_t num_args
);
```

This function:
1. Loads the kernel binary (`.vxbin` file)
2. Builds the device args buffer (header + user args)
3. Converts host buffer handles to 32-bit device addresses
4. Uploads args to device memory
5. Starts kernel execution

## Build System

### CMakeLists.txt
Located at `runtime/CMakeLists.txt`:
```cmake
add_library(vortex_hip SHARED
    hip_vortex_runtime/src/hip_device.cpp
    hip_vortex_runtime/src/hip_memory.cpp
    hip_vortex_runtime/src/hip_kernel.cpp
    hip_vortex_runtime/src/hip_error.cpp
)
```

### Build Commands
```bash
cd runtime/build
cmake ..
make
```

Output: `runtime/build/libhip_vortex.so`

## Dependencies

The HIP runtime depends on the Vortex runtime:

```
libhip_vortex.so
    └── libvortex.so (from vortex/build/runtime/)
        └── vx_dev_open, vx_mem_alloc, vx_start, etc.
```

## Recommendations for Cleanup

1. **Consolidate headers**: Merge `runtime/include/vortex_hip_runtime.h` with `runtime/hip_vortex_runtime/include/hip_vortex_runtime.h`

2. **Remove unused code**: The `runtime/src/vortex_hip_runtime.cpp` appears unused

3. **Simplify include structure**: Use a single `runtime/include/` directory

4. **Document the split**: Device headers (`hip_runtime_vortex/`) vs runtime library (`runtime/hip_vortex_runtime/`)

## File Reference

### Headers That Matter

| Header | Used By | Contains |
|--------|---------|----------|
| `runtime/hip_vortex_runtime/include/hip_vortex_runtime.h` | Host code | HIP API, dim3, VortexKernelArgMeta |
| `hip_runtime_vortex/hip_runtime.h` | Device code (Polygeist) | `__global__`, threadIdx, blockIdx |

### Source Files

| Source | Contains |
|--------|----------|
| `hip_vortex_runtime/src/hip_device.cpp` | Device management, Vortex device handle |
| `hip_vortex_runtime/src/hip_memory.cpp` | hipMalloc, hipMemcpy, buffer tracking |
| `hip_vortex_runtime/src/hip_kernel.cpp` | Kernel loading, vortexLaunchKernel |
| `hip_vortex_runtime/src/hip_error.cpp` | Error strings, last error tracking |

### Generated Files

| File | Generated By | Purpose |
|------|--------------|---------|
| `kernel_stubs.h` | generate_host_stubs.py | Type-safe kernel launchers |
| `*.meta.json` | ConvertGPUToVortex.cpp | Kernel argument metadata |
| `kernel.vxbin` | vxbin.py | Vortex kernel binary |
