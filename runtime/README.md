# Vortex HIP Runtime

A HIP (Heterogeneous-computing Interface for Portability) runtime implementation for the Vortex RISC-V GPU.

## Overview

This library provides a mapping from HIP API functions to Vortex GPU functions, enabling HIP/CUDA applications to run on Vortex hardware.

### Features

✅ **Runtime API (Host-Side)**
- Device management (`hipGetDeviceCount`, `hipSetDevice`, `hipGetDeviceProperties`)
- Memory management (`hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemset`)
- Kernel execution (`hipLaunchKernel`)
- Synchronization (`hipDeviceSynchronize`)
- Error handling (`hipGetLastError`, `hipGetErrorString`)

✅ **Device API (Kernel-Side)**
- Thread indexing (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`) - **Identical to Vortex!**
- Synchronization (`__syncthreads()`, `__syncthreads_count()`)
- Warp voting (`__all`, `__any`, `__ballot`)
- Warp shuffles (`__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor`)
- Atomic operations (`atomicAdd`, `atomicSub`, `atomicExch`, etc.)

## Directory Structure

```
runtime/
├── include/               # Public headers
│   ├── vortex_hip_runtime.h    # Host-side API
│   └── vortex_hip_device.h     # Device-side API
├── src/                   # Implementation
│   └── vortex_hip_runtime.cpp  # Runtime implementation
├── examples/              # Example programs
│   ├── vector_add.cpp           # Vector addition example
│   └── CMakeLists.txt
├── cmake/                 # CMake configuration
│   └── vortex_hip-config.cmake.in
├── CMakeLists.txt         # Main build configuration
└── README.md              # This file
```

## Building

### Prerequisites

1. **Vortex GPU** installed and built at `~/vortex`
2. **CMake** 3.10 or later
3. **C++14** compatible compiler

### Quick Build (Using Script)

```bash
cd ~/vortex_hip/runtime
./build.sh
```

### Manual Build Steps

```bash
# Set Vortex root (if not at ~/vortex)
export VORTEX_ROOT=~/vortex

# Ensure Vortex is built
cd $VORTEX_ROOT/build
make -j$(nproc)

# Build Vortex HIP Runtime
cd ~/vortex_hip/runtime
mkdir build && cd build
cmake .. -DVORTEX_ROOT=$VORTEX_ROOT -DCMAKE_INSTALL_PREFIX=/usr/local

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Build Options

- `BUILD_EXAMPLES=ON/OFF` - Build example programs (default: ON)
- `VORTEX_ROOT=<path>` - Path to Vortex installation (default: ~/vortex)
- `CMAKE_INSTALL_PREFIX=<path>` - Installation directory

### Hardware Testing

For detailed hardware testing instructions, see [HARDWARE_TESTING.md](HARDWARE_TESTING.md).

Quick test:
```bash
cd ~/vortex_hip/runtime
./test.sh
```

## Usage

### Including in Your Project

#### CMake

```cmake
find_package(vortex_hip REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp vortex_hip::vortex_hip)
```

#### Manual

```bash
# Compile
g++ -c myapp.cpp \
    -I/path/to/vortex_hip/include \
    -I$VORTEX_ROOT/runtime/include

# Link
g++ myapp.o \
    -L/path/to/vortex_hip/lib -lvortex_hip \
    -L$VORTEX_ROOT/stub -lvortex \
    -o myapp
```

### Example: Vector Addition

```cpp
#include "vortex_hip_runtime.h"

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Initialize
    hipInit(0);
    hipSetDevice(0);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // Copy to device
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel (requires kernel binary)
    // hipLaunchKernelGGL(vectorAdd,
    //                    dim3((N+255)/256), dim3(256),
    //                    0, 0,
    //                    d_a, d_b, d_c, N);

    // Synchronize
    hipDeviceSynchronize();

    // Copy result back
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at %d\n", i);
        }
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```

## API Documentation

### Runtime API (vortex_hip_runtime.h)

See the [API Mapping Documentation](../docs/implementation/HIP-TO-VORTEX-API-MAPPING.md) for complete details.

Key functions:

| Function | Description |
|----------|-------------|
| `hipInit(flags)` | Initialize HIP runtime |
| `hipGetDeviceCount(count)` | Get number of devices |
| `hipSetDevice(deviceId)` | Set active device |
| `hipGetDeviceProperties(prop, deviceId)` | Query device capabilities |
| `hipMalloc(ptr, size)` | Allocate device memory |
| `hipFree(ptr)` | Free device memory |
| `hipMemcpy(dst, src, size, kind)` | Copy memory |
| `hipLaunchKernel(...)` | Launch kernel |
| `hipDeviceSynchronize()` | Wait for device |

### Device API (vortex_hip_device.h)

Built-in variables (identical to Vortex!):
- `threadIdx.{x,y,z}` - Thread index within block
- `blockIdx.{x,y,z}` - Block index within grid
- `blockDim.{x,y,z}` - Block dimensions
- `gridDim.{x,y,z}` - Grid dimensions

Functions:
- `__syncthreads()` - Block-level barrier
- `__all(pred)`, `__any(pred)`, `__ballot(pred)` - Warp voting
- `__shfl*()` - Warp shuffle operations
- `atomic*()` - Atomic operations

## Key Mapping Details

### Thread Indexing

**IDENTICAL between HIP and Vortex!** No translation needed.

```cpp
// Works the same in HIP and Vortex
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

### Synchronization

```cpp
// HIP: __syncthreads()
// Vortex: __syncthreads() (maps to vx_barrier)
// Result: Identical usage!
```

### Warp Operations

```cpp
// HIP → Vortex mapping
__all(pred)      → vx_vote_all(pred)
__any(pred)      → vx_vote_any(pred)
__ballot(pred)   → vx_vote_ballot(pred)
__shfl_up(...)   → vx_shfl_up(...)
__shfl_down(...) → vx_shfl_down(...)
```

### Atomics

```cpp
// Implemented using RISC-V atomic instructions
atomicAdd(addr, val)  → amoadd.w instruction
atomicExch(addr, val) → amoswap.w instruction
atomicMin(addr, val)  → amomin.w instruction
```

## Limitations

### Current Status

This is an **initial implementation** with the following limitations:

1. **Kernel Compilation**: Requires separate kernel compilation pipeline (not yet implemented)
2. **Argument Marshalling**: Basic implementation; needs metadata for complex types
3. **Device-to-Device Copy**: Not yet supported
4. **Streams**: Basic stub; async operations not fully implemented
5. **Textures**: Not implemented
6. **Graphics Interop**: Not implemented

### Roadmap

**Phase 1** (Current):
- ✅ Basic runtime API
- ✅ Memory management
- ✅ Device queries
- ✅ Kernel launch infrastructure

**Phase 2** (Next):
- ⬜ Kernel compilation pipeline
- ⬜ Complete argument marshalling
- ⬜ Stream support
- ⬜ Events

**Phase 3** (Future):
- ⬜ Textures
- ⬜ Device-to-device copy
- ⬜ Multi-device support
- ⬜ Performance optimizations

## Implementation Strategy

This library follows the **Tier 2 approach** from the project documentation:

1. **Base Layer**: Use chipStar + OpenCL for standard HIP (90% coverage)
2. **Extension Layer**: Add Vortex-specific optimizations (this library)
3. **Optional Direct Backend**: Bypass OpenCL for maximum performance

## Testing

### Running Examples

```bash
# After building
cd build/examples
./vector_add
```

### Expected Output

```
Vortex HIP Vector Addition Example
===================================

Number of devices: 1

Device 0: Vortex RISC-V GPU
  Compute capability: 1.0
  Total global memory: 256 MB
  Shared memory per block: 4096 bytes
  Warp size: 4
  Max threads per block: 16
  Multiprocessor count: 4

Initializing host arrays with 1024 elements...
Allocating device memory...
Copying data to device...
Launching kernel...
...
```

## Troubleshooting

### Library Not Found

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$VORTEX_ROOT/stub:$LD_LIBRARY_PATH
```

### Vortex Headers Not Found

```bash
export VORTEX_ROOT=/path/to/vortex
cmake .. -DVORTEX_ROOT=$VORTEX_ROOT
```

### Link Errors

Ensure you link to **both** libraries:
```bash
-lvortex_hip -lvortex
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## References

- [HIP API Mapping Documentation](../docs/implementation/HIP-TO-VORTEX-API-MAPPING.md)
- [Vortex Architecture](../docs/reference/VORTEX-ARCHITECTURE.md)
- [Compiler Infrastructure](../docs/implementation/COMPILER_INFRASTRUCTURE.md)
- [Official HIP Documentation](https://rocm.docs.amd.com/projects/HIP/)

## License

Apache License 2.0 - See [LICENSE](../LICENSE) for details.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-05
**Status**: Initial implementation
