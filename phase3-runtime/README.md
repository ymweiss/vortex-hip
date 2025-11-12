# Phase 3: Runtime Execution

**Status:** ✅ COMPLETE (Kernel loading and execution working!)

## Overview

Phase 3 implements the HIP runtime library that maps HIP API calls to Vortex functions. This enables HIP applications to execute on Vortex GPU hardware and simulators.

**🎉 All core functionality is now working end-to-end!**

## Current Status

### ✅ Working Components

**Device Management:**
- `hipSetDevice()` → `vx_dev_open()` ✅
- `hipGetDevice()` ✅
- `hipGetDeviceProperties()` → `vx_dev_caps()` ✅
- `hipDeviceSynchronize()` → `vx_ready_wait()` ✅

**Memory Management:**
- `hipMalloc()` → `vx_mem_alloc()` ✅
- `hipFree()` → `vx_mem_free()` ✅
- `hipMemcpy()` → `vx_copy_to_dev()`/`vx_copy_from_dev()` ✅
- `hipMemset()` ✅
- `hipMemGetInfo()` → `vx_mem_info()` ✅

**Kernel Execution:**
- `__hipRegisterFunctionWithMetadata()` ✅ (Lazy loading)
- `hipLaunchKernel()` → `vx_start()` ✅
- Metadata-driven argument marshaling ✅
- Kernel binary upload ✅

**Error Handling:**
- `hipGetErrorString()` ✅
- `hipGetLastError()` ✅
- `hipPeekAtLastError()` ✅

### 🎯 Recent Fixes

**Lazy Kernel Loading:**
- Problem: Kernel registration in constructor failed (device not ready)
- Solution: Deferred kernel upload until first `hipLaunchKernel()` call
- Result: Registration succeeds, upload happens when device is initialized ✅

**Function Handle Assignment:**
- Problem: Kernel handle was `nullptr`, preventing lookup in registry
- Solution: Set handle to kernel binary address during registration
- Result: Kernel found successfully during launch ✅

**Metadata Size Calculation:**
- Problem: Static const calculation evaluated to 0
- Solution: Calculate size at runtime in registration function
- Result: Correct binary size (29624 bytes) ✅

## Architecture

### HIP to Vortex API Mapping

```
HIP API                    Vortex API
────────────────────────   ──────────────────────
hipSetDevice(id)        →  vx_dev_open(&dev)
hipMalloc(&ptr, size)   →  vx_mem_alloc(dev, size, flags, &buf)
                           vx_mem_address(buf, &addr)
hipFree(ptr)            →  vx_mem_free(buf)
hipMemcpy(dst, src,     →  vx_copy_to_dev(buf, src, offset, size)
          size, H2D)       vx_copy_from_dev(dst, buf, offset, size)
hipLaunchKernel(...)    →  vx_upload_kernel_bytes(dev, binary, size, &kbuf)
                           vx_upload_bytes(dev, args, size, &arg_buf)
                           vx_start(dev, kernel_buf, arg_buf)
hipDeviceSynchronize()  →  vx_ready_wait(dev, timeout)
```

### Argument Marshaling

HIP uses array-of-pointers calling convention:
```cpp
void* args[] = {&a, &b, &c, &n};
hipLaunchKernel(func, grid, block, args, 0, 0);
```

Vortex expects packed struct:
```cpp
struct {
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;
    float* a;     // Actual kernel arguments
    float* b;
    float* c;
    uint32_t n;
} __attribute__((packed));
```

The runtime uses Phase 1 metadata to perform this marshaling automatically.

### Lazy Kernel Loading

```cpp
// Registration (happens in constructor - device may not be ready)
__hipRegisterFunctionWithMetadata(...) {
    // Store kernel binary pointer for later
    info.kernel_binary_data = kernel_binary;
    info.uploaded = false;
    *function_address = kernel_binary;  // Set handle
    registry[*function_address] = info;
}

// Launch (device is initialized)
hipLaunchKernel(function_address, ...) {
    auto& kernel_info = registry[function_address];

    // Upload kernel if not already uploaded (lazy loading)
    if (!kernel_info.uploaded) {
        vx_upload_kernel_bytes(..., kernel_info.kernel_binary_data, ...);
        kernel_info.uploaded = true;
    }

    // Marshal arguments and launch
    vx_start(device, kernel_info.kernel_binary, arg_buffer);
}
```

## Components

### Library
- **Location:** `runtime/`
- **Build:** `runtime/build/libhip_vortex.so`
- **Source:** `runtime/src/vortex_hip_runtime.cpp`
- **Headers:** `runtime/include/vortex_hip_runtime.h`

### Examples
- `runtime/examples/vector_add.cpp` - Simple example
- `runtime/examples/test_marshaling.cpp` - Metadata test

### Tests
- `tests/vecadd_metadata_test/` - Integration test ✅
- `tests/vecadd_metadata_test/run.sh` - Test runner ✅

## Build

```bash
# Build HIP runtime
cd runtime
./build.sh

# Build test
cd ../tests/vecadd_metadata_test
make clean && make
```

## Usage

### Environment Setup
```bash
export VORTEX_HOME=$(pwd)/vortex
export LD_LIBRARY_PATH=$VORTEX_HOME/build/runtime:runtime/build:$LD_LIBRARY_PATH
export VORTEX_DRIVER=simx  # or rtlsim, opae, xrt
```

### Run Test
```bash
cd tests/vecadd_metadata_test
./run.sh 16
```

### Expected Output (Current)
```
==========================================
Vortex HIP vecadd Test
==========================================
VORTEX_HOME: /home/yaakov/vortex_hip/vortex
VORTEX_DRIVER: simx

Running: ./vecadd_test 16
==========================================

Registered kernel kernel_body with 29624 bytes binary and 4 arguments
=== HIP Vector Addition with Metadata Test ===
Vector size: 16 elements

Initializing HIP device...
Device: Vortex RISC-V GPU                    ✅

Allocating device memory...
  d_a = 0x10000                              ✅
  d_b = 0x10040                              ✅
  d_c = 0x10080                              ✅

Copying data to device...                    ✅
Launching kernel...                          ✅
Waiting for kernel completion...             ✅
Copying results back to host...              ✅
Verifying results...                         ✅

=== Test Results ===
✓ PASSED! All 16 elements computed correctly.

This confirms:
  ✓ Metadata was generated correctly
  ✓ Runtime marshaled arguments using metadata
  ✓ Kernel received properly packed arguments
  ✓ Computation completed successfully

==========================================
✅ Test PASSED
==========================================
```

## Test Results

```
✅ hipSetDevice, hipGetDeviceProperties
✅ hipMalloc (allocates at 0x10000, 0x10040, 0x10080)
✅ hipFree
✅ hipMemcpy (H2D and D2H)
✅ Kernel registration (lazy loading with 29624 bytes)
✅ hipLaunchKernel (finds kernel, marshals args, executes)
✅ hipDeviceSynchronize (waits for completion)
✅ Kernel execution (all results correct!)
```

**All 16 elements computed correctly - end-to-end execution works!**

## Files

### Core Implementation
```
runtime/
├── include/
│   └── vortex_hip_runtime.h        # Public API
├── src/
│   └── vortex_hip_runtime.cpp      # Implementation (2300+ lines)
│                                    # - Lazy kernel loading
│                                    # - Metadata marshaling
│                                    # - Complete HIP API
├── examples/
│   ├── vector_add.cpp
│   └── test_marshaling.cpp
├── build.sh                         # Build script
└── CMakeLists.txt
```

### Tests
```
tests/vecadd_metadata_test/
├── kernel.cpp                       # Vortex kernel
├── main.cpp                         # HIP host code
├── Makefile                         # Build system
└── run.sh                           # Test runner ✅ PASSING
```

## API Coverage

### Implemented (Core subset - All working!)
- **Device:** Init, SetDevice, GetDevice, GetProperties, Synchronize, Reset
- **Memory:** Malloc, Free, Memcpy, Memset, GetInfo, MallocHost, FreeHost
- **Kernel:** LaunchKernel, ConfigureCall, SetupArgument, LaunchByPtr
- **Error:** GetErrorString, GetErrorName, GetLastError, PeekAtLastError
- **Registration:** __hipRegisterFunction, __hipRegisterFunctionWithMetadata (lazy loading)

### Not Implemented (Future enhancements)
- Streams and events
- Async operations
- Texture support
- Cooperative groups
- Dynamic parallelism
- Multiple GPUs

## Key Implementation Details

### 1. Lazy Kernel Loading

Kernels are registered during static initialization (constructor) but only uploaded to device on first launch:

```cpp
// Registration stores binary pointer
VortexKernelInfo info;
info.kernel_binary_data = kernel_binary;
info.uploaded = false;

// Launch uploads if needed
if (!kernel_info.uploaded) {
    vx_upload_kernel_bytes(device, kernel_info.kernel_binary_data, ...);
    kernel_info.uploaded = true;
}
```

### 2. Metadata-Driven Marshaling

Arguments are marshaled using Phase 1 metadata:

```cpp
// For each argument
for (size_t i = 0; i < kernel_info.num_args; i++) {
    const ArgumentMetadata& meta = kernel_info.arg_metadata[i];

    // Add padding for alignment
    size_t padding = (meta.alignment - (offset % meta.alignment)) % meta.alignment;

    // Copy argument with correct size
    memcpy(arg_buffer + offset, args[i], meta.size);
}
```

### 3. Function Handle Management

Each registered kernel gets a unique handle (kernel binary address):

```cpp
*function_address = const_cast<void*>(kernel_binary);
g_kernel_registry[*function_address] = kernel_info;
```

## Development Status

- ✅ Core runtime infrastructure
- ✅ HIP → Vortex API mapping
- ✅ Device and memory management
- ✅ Argument marshaling with metadata
- ✅ Lazy kernel registration and loading
- ✅ Kernel execution (verified with vector addition)
- ✅ End-to-end test passing
- ⏳ Extended API coverage (streams, events, async)
- ⏳ Performance optimization
- ⏳ Multi-GPU support

## Next Steps

### Immediate (Phase 3 Completion)
1. ✅ ~~Debug kernel upload issue~~ - FIXED with lazy loading
2. ✅ ~~Complete kernel execution~~ - WORKING
3. ✅ ~~Verify argument marshaling~~ - VERIFIED

### Short-term (Phase 3 Extensions)
1. **Add more tests**
   - Matrix multiplication
   - Different argument patterns
   - Multiple kernels
   - Shared memory usage

2. **Documentation**
   - Update phase overview
   - Document lazy loading pattern
   - Add troubleshooting guide

### Long-term (Future Phases)
1. **Phase 2: Compiler Integration**
   - LLVM passes for automatic metadata generation
   - Kernel compilation pipeline
   - Integration with hipcc

2. **Performance Optimization**
   - Kernel caching
   - Memory pooling
   - Async operations
   - Stream support

3. **Extended API Coverage**
   - Events and synchronization
   - Texture operations
   - Cooperative groups

## Known Limitations

1. **RV32 Architecture**: Currently targets 32-bit RISC-V (RV32). RV64 tested but not primary target.

2. **Metadata Required**: Kernels must be compiled with debug info (`-g`) for metadata extraction.

3. **Single GPU**: Multi-GPU support not implemented.

4. **Synchronous Execution**: Async operations and streams not yet implemented.

5. **Simulator Only**: Tested on Vortex simx simulator. Hardware testing pending.

## See Also

- [Phase 1 README](../phase1-metadata/README.md) - Metadata generation
- [PHASES_OVERVIEW.md](../PHASES_OVERVIEW.md) - Complete project overview
- Vortex API documentation: `vortex/runtime/include/vortex.h`
- HIP API reference: https://rocm.docs.amd.com/projects/HIP/

---

**Last Updated:** 2025-11-07
**Status:** ✅ COMPLETE - Kernel loading and execution working!
