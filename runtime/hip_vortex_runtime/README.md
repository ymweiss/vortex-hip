# HIP Vortex Runtime Library

Minimal HIP runtime library that maps HIP API calls to Vortex API for the split compilation model.

## Overview

This library enables HIP host code to run on Vortex by providing HIP API implementations that delegate to the Vortex runtime (`libvortex.so`).

```
┌─────────────────────────────────────┐
│         HIP Host Code               │
│   hipMalloc(), hipLaunchKernel()    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    libhip_vortex_runtime.a          │  ← This library
│    Maps HIP API → Vortex API        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         libvortex.so                │
│    Vortex Runtime (SimX/FPGA)       │
└─────────────────────────────────────┘
```

## Build

```bash
# Set VORTEX_ROOT if not in default location
export VORTEX_ROOT=/path/to/vortex

# Build the library
make

# Output: lib/libhip_vortex_runtime.a
```

## Usage

### Compiling Host Code

```bash
g++ -std=c++17 \
    -I/path/to/hip_vortex_runtime/include \
    -I$VORTEX_ROOT/runtime/include \
    -I$VORTEX_ROOT/build/hw \
    your_host_code.cpp \
    -L/path/to/hip_vortex_runtime/lib \
    -L$VORTEX_ROOT/build/runtime \
    -lhip_vortex_runtime -lvortex \
    -o your_program
```

### Running

```bash
# Set library path for libvortex.so
export LD_LIBRARY_PATH=$VORTEX_ROOT/build/runtime:$LD_LIBRARY_PATH

# Set kernel path (optional, defaults to current directory)
export VORTEX_KERNEL_PATH=/path/to/kernels/

./your_program
```

## Implemented APIs

### Memory Management

| HIP API | Vortex Mapping | Status |
|---------|----------------|--------|
| `hipMalloc()` | `vx_mem_alloc()` | ✅ |
| `hipFree()` | `vx_mem_free()` | ✅ |
| `hipMemcpy()` | `vx_copy_to_dev()` / `vx_copy_from_dev()` | ✅ |
| `hipMemset()` | `vx_copy_to_dev()` (with filled buffer) | ✅ |

### Device Management

| HIP API | Vortex Mapping | Status |
|---------|----------------|--------|
| `hipInit()` | (no-op, lazy init) | ✅ |
| `hipSetDevice()` | `vx_dev_open()` | ✅ |
| `hipGetDevice()` | (returns current device) | ✅ |
| `hipGetDeviceCount()` | (returns 1) | ✅ |
| `hipGetDeviceProperties()` | `vx_dev_caps()` | ✅ |
| `hipDeviceSynchronize()` | `vx_ready_wait()` | ✅ |
| `hipDeviceReset()` | `vx_dev_close()` | ✅ |

### Error Handling

| HIP API | Status |
|---------|--------|
| `hipGetErrorString()` | ✅ |
| `hipGetErrorName()` | ✅ |
| `hipGetLastError()` | ✅ |
| `hipPeekAtLastError()` | ✅ |

### Kernel Launch

| HIP API | Vortex Mapping | Status |
|---------|----------------|--------|
| `hipRegisterKernel()` | `vx_upload_kernel_file()` | ✅ |
| `hipLaunchKernelGGL()` | `vx_start()` | ✅ |
| `hipModuleLoad()` | `vx_upload_kernel_file()` | ✅ |
| `hipModuleUnload()` | `vx_mem_free()` | ✅ |
| `hipModuleGetFunction()` | (returns module handle) | ✅ |
| `hipModuleLaunchKernel()` | `vx_start()` | ✅ |

## Kernel Registration

Before launching a kernel, register it with the runtime:

```cpp
// Register kernel binary
hipRegisterKernel("my_kernel", "my_kernel.vxbin");

// Or use VORTEX_KERNEL_PATH environment variable
// Kernels are auto-loaded from: $VORTEX_KERNEL_PATH/<kernel_name>.vxbin
```

## Example

```cpp
#include <hip/hip_runtime.h>
#include <vector>

// Kernel declaration (compiled separately)
__global__ void vecadd(float* a, float* b, float* c, int n);

int main() {
    const int N = 1024;
    std::vector<float> h_a(N), h_b(N), h_c(N);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc((void**)&d_a, N * sizeof(float));
    hipMalloc((void**)&d_b, N * sizeof(float));
    hipMalloc((void**)&d_c, N * sizeof(float));

    // Copy to device
    hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // Register and launch kernel
    hipRegisterKernel("vecadd", "vecadd.vxbin");
    hipLaunchKernelGGL(vecadd, dim3(N/64), dim3(64), 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    // Copy results back
    hipMemcpy(h_c.data(), d_c, N * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
```

## Tests

```bash
cd test

# Build tests
make

# Run native kernel test (uses pre-built Vortex vecadd kernel)
make run-native
```

## Architecture Notes

### Buffer Handles vs Device Pointers

HIP uses `void*` device pointers, but Vortex uses opaque `vx_buffer_h` handles. The runtime stores buffer handles as device pointers:

```cpp
hipError_t hipMalloc(void** devPtr, size_t size) {
    vx_buffer_h buffer;
    vx_mem_alloc(device, size, 0, &buffer);
    *devPtr = (void*)buffer;  // Store handle as pointer
}
```

When passing to kernels, the runtime converts handles to device addresses using `vx_mem_address()`.

### Kernel Argument Passing

Vortex kernels receive arguments through a packed structure uploaded to device memory. The runtime packs HIP kernel arguments into this format:

```cpp
struct VortexKernelArgs {
    uint32_t grid_dim[3];   // Grid dimensions
    uint32_t block_dim[3];  // Block dimensions
    // User arguments follow...
};
```

### TODO: Kernel Metadata

The runtime currently requires manual kernel registration. Future work will add automatic metadata parsing from JSON companion files emitted during kernel compilation:

```json
{
  "kernel_name": "vecadd",
  "arguments": [
    {"name": "a", "type": "ptr", "size": 8},
    {"name": "b", "type": "ptr", "size": 8},
    {"name": "c", "type": "ptr", "size": 8},
    {"name": "n", "type": "i32", "size": 4}
  ]
}
```

## Files

```
hip_vortex_runtime/
├── include/
│   ├── hip_vortex_runtime.h   # Main header with all HIP API declarations
│   └── hip/
│       └── hip_runtime.h      # Standard HIP include path wrapper
├── src/
│   ├── hip_memory.cpp         # Memory management (hipMalloc, hipMemcpy, etc.)
│   ├── hip_device.cpp         # Device management (hipSetDevice, hipGetDeviceProperties)
│   ├── hip_error.cpp          # Error handling (hipGetErrorString, hipGetLastError)
│   └── hip_kernel.cpp         # Kernel launch (hipLaunchKernelGGL, hipModuleLoad)
├── test/
│   ├── test_native_kernel.cpp # Test with native Vortex kernel
│   └── Makefile
├── Makefile
└── README.md
```
