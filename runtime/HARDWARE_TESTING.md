# Hardware Testing Guide

This guide explains how to build and test the Vortex HIP runtime on actual Vortex hardware.

## Prerequisites

### 1. Vortex Installation

Vortex must be built and installed at `~/vortex`:

```bash
# Check Vortex is present
ls ~/vortex/runtime/include/vortex.h
ls ~/vortex/build/runtime/libvortex.so
```

If Vortex is not built:

```bash
cd ~/vortex
# Follow Vortex build instructions
# Typically:
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### 2. Environment Setup

```bash
# Set Vortex root (should point to ~/vortex)
export VORTEX_ROOT=~/vortex

# Add Vortex library to library path
export LD_LIBRARY_PATH=$VORTEX_ROOT/build/runtime:$LD_LIBRARY_PATH

# Source Vortex toolchain (if needed)
source $VORTEX_ROOT/build/ci/toolchain_env.sh
```

## Building Vortex HIP Runtime

### Step 1: Configure

```bash
cd ~/vortex_hip/runtime
mkdir -p build
cd build

# Configure with Vortex
cmake .. \
    -DVORTEX_ROOT=~/vortex \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

**Expected Output:**
```
-- Vortex root: /home/yaakov/vortex
-- Vortex include dir: /home/yaakov/vortex/runtime/include
-- Vortex kernel include dir: /home/yaakov/vortex/kernel/include
-- Vortex library: /home/yaakov/vortex/build/runtime/libvortex.so
-- Configuring done
-- Generating done
```

If you see errors about missing Vortex components, check:
1. `VORTEX_ROOT` is set correctly
2. Vortex is built (`~/vortex/build/runtime/libvortex.so` exists)
3. Headers are present (`~/vortex/runtime/include/vortex.h` exists)

### Step 2: Build

```bash
make -j$(nproc)
```

**Expected Output:**
```
[ 50%] Building CXX object CMakeFiles/vortex_hip.dir/src/vortex_hip_runtime.cpp.o
[100%] Linking CXX shared library libhip_vortex.so
[100%] Built target vortex_hip
[100%] Built target vector_add
```

### Step 3: Verify Build

```bash
# Check library was created
ls -lh libhip_vortex.so

# Check examples were built
ls -lh examples/vector_add

# Verify library dependencies
ldd libhip_vortex.so | grep vortex
```

**Expected:**
```
libhip_vortex.so -> libvortex.so
```

## Running Tests

### Test 1: Basic Device Query

Run the vector_add example which includes device queries:

```bash
cd ~/vortex_hip/runtime/build
./examples/vector_add
```

**Expected Output:**
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

### Test 2: Memory Operations

Create a simple test program:

```bash
cat > test_memory.cpp << 'EOF'
#include "vortex_hip_runtime.h"
#include <stdio.h>

#define HIP_CHECK(cmd) do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
        fprintf(stderr, "Error: %s at %s:%d\n", \
                hipGetErrorString(error), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

int main() {
    printf("Testing Vortex HIP Memory Operations\n");
    printf("=====================================\n\n");

    // Initialize
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(0));

    // Get memory info
    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    printf("Device Memory:\n");
    printf("  Free:  %zu MB\n", free_mem / (1024 * 1024));
    printf("  Total: %zu MB\n\n", total_mem / (1024 * 1024));

    // Allocate device memory
    const size_t size = 1024 * 1024 * sizeof(float); // 4 MB
    printf("Allocating 4 MB on device...\n");
    float* d_ptr;
    HIP_CHECK(hipMalloc((void**)&d_ptr, size));
    printf("  Success! Device pointer: %p\n\n", d_ptr);

    // Allocate host memory
    printf("Allocating 4 MB on host...\n");
    float* h_ptr = (float*)malloc(size);
    if (!h_ptr) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }
    printf("  Success! Host pointer: %p\n\n", h_ptr);

    // Initialize host memory
    printf("Initializing host data...\n");
    for (size_t i = 0; i < size / sizeof(float); i++) {
        h_ptr[i] = (float)i;
    }
    printf("  Done (first value: %f, last value: %f)\n\n",
           h_ptr[0], h_ptr[size / sizeof(float) - 1]);

    // Copy to device
    printf("Copying 4 MB to device...\n");
    HIP_CHECK(hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice));
    printf("  Success!\n\n");

    // Memset on device
    printf("Setting device memory to zero...\n");
    HIP_CHECK(hipMemset(d_ptr, 0, size));
    printf("  Success!\n\n");

    // Copy back
    printf("Copying 4 MB from device...\n");
    HIP_CHECK(hipMemcpy(h_ptr, d_ptr, size, hipMemcpyDeviceToHost));
    printf("  Success!\n\n");

    // Verify
    printf("Verifying data (should be all zeros)...\n");
    bool all_zero = true;
    for (size_t i = 0; i < size / sizeof(float); i++) {
        if (h_ptr[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    printf("  %s\n\n", all_zero ? "PASS" : "FAIL");

    // Cleanup
    printf("Cleaning up...\n");
    HIP_CHECK(hipFree(d_ptr));
    free(h_ptr);
    printf("  Done!\n\n");

    printf("All tests passed!\n");
    return 0;
}
EOF

# Compile
g++ test_memory.cpp \
    -I../include \
    -I$VORTEX_ROOT/runtime/include \
    -L. -lhip_vortex \
    -L$VORTEX_ROOT/build/runtime -lvortex \
    -Wl,-rpath,.:$VORTEX_ROOT/build/runtime \
    -o test_memory

# Run
./test_memory
```

### Test 3: Device Properties

```bash
cat > test_properties.cpp << 'EOF'
#include "vortex_hip_runtime.h"
#include <stdio.h>

int main() {
    hipInit(0);

    int count;
    hipGetDeviceCount(&count);
    printf("Found %d device(s)\n\n", count);

    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Architecture: %s\n", prop.gcnArchName);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("  Shared Memory Per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Dimensions: [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("  Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  PCI Bus ID: %d\n", prop.pciBusID);
        printf("\n");
    }

    return 0;
}
EOF

g++ test_properties.cpp \
    -I../include \
    -I$VORTEX_ROOT/runtime/include \
    -L. -lhip_vortex \
    -L$VORTEX_ROOT/build/runtime -lvortex \
    -Wl,-rpath,.:$VORTEX_ROOT/build/runtime \
    -o test_properties

./test_properties
```

## Troubleshooting

### Problem: Library Not Found

**Error:**
```
error while loading shared libraries: libhip_vortex.so: cannot open shared object file
```

**Solution:**
```bash
export LD_LIBRARY_PATH=~/vortex_hip/runtime/build:$LD_LIBRARY_PATH
```

Or use `-Wl,-rpath` when compiling:
```bash
g++ ... -Wl,-rpath,/path/to/vortex_hip/runtime/build
```

### Problem: Vortex Library Not Found

**Error:**
```
error while loading shared libraries: libvortex.so: cannot open shared object file
```

**Solution:**
```bash
export LD_LIBRARY_PATH=$VORTEX_ROOT/build/runtime:$LD_LIBRARY_PATH
```

### Problem: No Devices Found

**Error:**
```
Number of devices: 0
```

**Causes:**
1. Vortex hardware not connected/detected
2. Vortex driver not loaded
3. Permissions issue

**Debug:**
```bash
# Check if Vortex device is accessible
ls -l /dev/vortex* 2>/dev/null

# Test with native Vortex tools
cd $VORTEX_ROOT/tests/runtime/vecadd
make
./vecadd
```

### Problem: Memory Allocation Fails

**Error:**
```
Error: hipErrorOutOfMemory
```

**Debug:**
```bash
# Check available memory
cd ~/vortex_hip/runtime/build
./test_properties
# Look at "Total Global Memory"

# Try smaller allocation
# Reduce size in test program
```

### Problem: Synchronization Timeout

**Error:**
```
Error: hipErrorUnknown at vortex_hip_runtime.cpp:xxx
```

**Causes:**
1. Kernel didn't complete
2. Hardware issue
3. Timeout too short

**Debug:**
```bash
# Check Vortex logs
dmesg | tail -50

# Increase timeout (in code)
vx_ready_wait(device, VX_MAX_TIMEOUT);
```

## Performance Testing

### Memory Bandwidth

```bash
cat > bandwidth_test.cpp << 'EOF'
#include "vortex_hip_runtime.h"
#include <stdio.h>
#include <chrono>

using namespace std::chrono;

int main() {
    hipInit(0);
    hipSetDevice(0);

    // Test various sizes
    const size_t sizes[] = {
        1 << 10,   // 1 KB
        1 << 20,   // 1 MB
        1 << 24,   // 16 MB
    };

    printf("Memory Bandwidth Test\n");
    printf("=====================\n\n");

    for (size_t size : sizes) {
        void* d_ptr;
        void* h_ptr = malloc(size);
        hipMalloc(&d_ptr, size);

        // Host to Device
        auto start = high_resolution_clock::now();
        hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        auto end = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        double gbps = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

        printf("Size: %zu KB\n", size / 1024);
        printf("  H2D: %.2f ms, %.2f GB/s\n", ms, gbps);

        // Device to Host
        start = high_resolution_clock::now();
        hipMemcpy(h_ptr, d_ptr, size, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        end = high_resolution_clock::now();

        ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        gbps = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

        printf("  D2H: %.2f ms, %.2f GB/s\n\n", ms, gbps);

        hipFree(d_ptr);
        free(h_ptr);
    }

    return 0;
}
EOF

g++ bandwidth_test.cpp \
    -I../include \
    -I$VORTEX_ROOT/runtime/include \
    -L. -lhip_vortex \
    -L$VORTEX_ROOT/build/runtime -lvortex \
    -Wl,-rpath,.:$VORTEX_ROOT/build/runtime \
    -o bandwidth_test

./bandwidth_test
```

## Expected Results

### On SimX (Simulator)

- **Device Count**: 1
- **Memory**: Depends on configuration (typically 256MB-1GB)
- **Warp Size**: 4-32 (configurable)
- **Memory Bandwidth**: ~1-10 MB/s (simulation overhead)

### On RTL Simulation

- **Device Count**: 1
- **Memory**: Depends on configuration
- **Warp Size**: Matches hardware config
- **Memory Bandwidth**: Very slow (simulation)

### On FPGA

- **Device Count**: 1
- **Memory**: Actual hardware memory
- **Warp Size**: Hardware configuration
- **Memory Bandwidth**: ~100 MB/s to 1 GB/s (depends on FPGA)

## Next Steps

Once basic tests pass:

1. **Test Kernel Launch** - Need kernel compilation pipeline
2. **Benchmark Performance** - Compare with native Vortex
3. **Run HIP Applications** - Port existing HIP codes
4. **Optimize** - Profile and improve bottlenecks

## Support

If you encounter issues:

1. Check [README.md](README.md) for general documentation
2. See [PHASES_OVERVIEW.md](../PHASES_OVERVIEW.md) for current project status
3. Review Vortex documentation at `$VORTEX_ROOT/README.md`
4. File issues with detailed error messages and system info

---

**Last Updated**: 2025-11-05
**Status**: Hardware testing procedures defined
**Tested On**: SimX simulator, FPGA pending
