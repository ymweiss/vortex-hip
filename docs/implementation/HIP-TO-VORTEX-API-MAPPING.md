# HIP to Vortex API Mapping

## Overview

This document provides a comprehensive mapping between HIP API functions and Vortex GPU functions. This mapping is essential for implementing HIP support on Vortex, whether through:
1. **Tier 1**: chipStar + OpenCL (use as reference)
2. **Tier 2**: Vortex-specific extensions
3. **Tier 3**: Direct Vortex backend implementation

---

## Table of Contents

1. [Runtime API Mappings (Host-Side)](#runtime-api-mappings-host-side)
2. [Kernel API Mappings (Device-Side)](#kernel-api-mappings-device-side)
3. [Thread Indexing](#thread-indexing)
4. [Synchronization Primitives](#synchronization-primitives)
5. [Memory Management](#memory-management)
6. [Warp/Wavefront Operations](#warpwavefront-operations)
7. [Implementation Templates](#implementation-templates)

---

## Runtime API Mappings (Host-Side)

### Device Management

| HIP API | Vortex API | Mapping Details |
|---------|------------|-----------------|
| `hipInit(unsigned int flags)` | `vx_dev_open(vx_device_h*)` | Initialize runtime; flags=0 for HIP, open device for Vortex |
| `hipGetDeviceCount(int* count)` | Check `vx_dev_open()` return | Vortex typically single device; return 1 if open succeeds |
| `hipSetDevice(int deviceId)` | `vx_dev_open(&device)` | Open device handle; deviceId typically 0 for Vortex |
| `hipGetDevice(int* deviceId)` | Store current device ID | Track current device in runtime state |
| `hipDeviceSynchronize()` | `vx_ready_wait(device, timeout)` | Wait for all device operations to complete |

**Implementation Template**:
```cpp
// Global state
static vx_device_h g_vortex_device = nullptr;
static int g_current_device = -1;

hipError_t hipInit(unsigned int flags) {
    if (flags != 0) return hipErrorInvalidValue;
    return hipSuccess;  // Lazy initialization on first use
}

hipError_t hipGetDeviceCount(int* count) {
    if (!count) return hipErrorInvalidValue;
    // Try to open device to check availability
    vx_device_h device;
    if (vx_dev_open(&device) == 0) {
        vx_dev_close(device);
        *count = 1;
        return hipSuccess;
    }
    *count = 0;
    return hipSuccess;
}

hipError_t hipSetDevice(int deviceId) {
    if (deviceId != 0) return hipErrorInvalidDevice;

    if (g_vortex_device == nullptr) {
        int result = vx_dev_open(&g_vortex_device);
        if (result != 0) return hipErrorInitializationError;
    }

    g_current_device = deviceId;
    return hipSuccess;
}

hipError_t hipDeviceSynchronize() {
    if (g_vortex_device == nullptr) return hipErrorInitializationError;

    int result = vx_ready_wait(g_vortex_device, VX_MAX_TIMEOUT);
    return (result == 0) ? hipSuccess : hipErrorUnknown;
}
```

### Device Properties

| HIP Property | Vortex Capability | Query Method |
|--------------|-------------------|--------------|
| `hipDeviceProp_t::name` | "Vortex RISC-V GPU" | Static string |
| `hipDeviceProp_t::totalGlobalMem` | `VX_CAPS_GLOBAL_MEM_SIZE` | `vx_dev_caps(device, VX_CAPS_GLOBAL_MEM_SIZE, &value)` |
| `hipDeviceProp_t::sharedMemPerBlock` | `VX_CAPS_LOCAL_MEM_SIZE` | `vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE, &value)` |
| `hipDeviceProp_t::maxThreadsPerBlock` | `threads * warps` | Query `VX_CAPS_NUM_THREADS` × `VX_CAPS_NUM_WARPS` |
| `hipDeviceProp_t::multiProcessorCount` | `VX_CAPS_NUM_CORES` | `vx_dev_caps(device, VX_CAPS_NUM_CORES, &value)` |
| `hipDeviceProp_t::warpSize` | `VX_CAPS_NUM_THREADS` | `vx_dev_caps(device, VX_CAPS_NUM_THREADS, &value)` |
| `hipDeviceProp_t::major` | 1 | Static (define Vortex compute capability) |
| `hipDeviceProp_t::minor` | 0 | Static |

**Implementation Template**:
```cpp
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    if (!prop) return hipErrorInvalidValue;
    if (deviceId != 0) return hipErrorInvalidDevice;
    if (g_vortex_device == nullptr) {
        hipSetDevice(0);  // Initialize if needed
    }

    // Clear structure
    memset(prop, 0, sizeof(hipDeviceProp_t));

    // Static properties
    strcpy(prop->name, "Vortex RISC-V GPU");
    prop->major = 1;
    prop->minor = 0;

    // Query hardware capabilities
    uint64_t value;

    vx_dev_caps(g_vortex_device, VX_CAPS_GLOBAL_MEM_SIZE, &value);
    prop->totalGlobalMem = value;

    vx_dev_caps(g_vortex_device, VX_CAPS_LOCAL_MEM_SIZE, &value);
    prop->sharedMemPerBlock = value;

    vx_dev_caps(g_vortex_device, VX_CAPS_NUM_CORES, &value);
    prop->multiProcessorCount = value;

    vx_dev_caps(g_vortex_device, VX_CAPS_NUM_THREADS, &value);
    prop->warpSize = value;

    uint64_t num_threads, num_warps;
    vx_dev_caps(g_vortex_device, VX_CAPS_NUM_THREADS, &num_threads);
    vx_dev_caps(g_vortex_device, VX_CAPS_NUM_WARPS, &num_warps);
    prop->maxThreadsPerBlock = num_threads * num_warps;

    // Max dimensions (conservative defaults)
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;

    prop->maxGridSize[0] = 65535;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;

    return hipSuccess;
}
```

---

## Memory Management

### Memory Allocation

| HIP API | Vortex API | Notes |
|---------|------------|-------|
| `hipMalloc(void** ptr, size_t size)` | `vx_mem_alloc(device, size, VX_MEM_READ_WRITE, &buffer)` | Allocate device memory |
| `hipFree(void* ptr)` | `vx_mem_free(buffer)` | Free device memory |
| `hipMallocHost(void** ptr, size_t size)` | `malloc(size)` | Regular host allocation (pinning not critical for Vortex) |
| `hipFreeHost(void* ptr)` | `free(ptr)` | Free host memory |

**Implementation Template**:
```cpp
// Track allocations
struct VortexAllocation {
    vx_buffer_h buffer;
    uint64_t device_addr;
    size_t size;
};

static std::unordered_map<void*, VortexAllocation> g_allocations;

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) return hipErrorInvalidValue;
    if (size == 0) {
        *ptr = nullptr;
        return hipSuccess;
    }

    if (g_vortex_device == nullptr) {
        hipSetDevice(0);  // Initialize
    }

    vx_buffer_h buffer;
    int result = vx_mem_alloc(g_vortex_device, size, VX_MEM_READ_WRITE, &buffer);
    if (result != 0) return hipErrorOutOfMemory;

    // Get device address
    uint64_t dev_addr;
    vx_mem_address(buffer, &dev_addr);

    // Store mapping (use device address as host pointer)
    void* host_ptr = reinterpret_cast<void*>(dev_addr);
    g_allocations[host_ptr] = {buffer, dev_addr, size};

    *ptr = host_ptr;
    return hipSuccess;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) return hipSuccess;

    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) return hipErrorInvalidValue;

    int result = vx_mem_free(it->second.buffer);
    g_allocations.erase(it);

    return (result == 0) ? hipSuccess : hipErrorInvalidValue;
}
```

### Memory Transfer

| HIP API | Vortex API | Direction |
|---------|------------|-----------|
| `hipMemcpy(dst, src, size, hipMemcpyHostToDevice)` | `vx_copy_to_dev(buffer, src, 0, size)` | Host → Device |
| `hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)` | `vx_copy_from_dev(dst, buffer, 0, size)` | Device → Host |
| `hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice)` | Device-side copy (via kernel or DMA) | Device → Device |
| `hipMemset(ptr, value, size)` | `vx_copy_to_dev()` with temp buffer | Set memory to value |

**Implementation Template**:
```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
    if (!dst || !src) return hipErrorInvalidValue;
    if (sizeBytes == 0) return hipSuccess;

    int result = 0;

    switch (kind) {
    case hipMemcpyHostToDevice: {
        auto it = g_allocations.find(dst);
        if (it == g_allocations.end()) return hipErrorInvalidValue;
        result = vx_copy_to_dev(it->second.buffer, src, 0, sizeBytes);
        break;
    }
    case hipMemcpyDeviceToHost: {
        auto it = g_allocations.find(const_cast<void*>(src));
        if (it == g_allocations.end()) return hipErrorInvalidValue;
        result = vx_copy_from_dev(dst, it->second.buffer, 0, sizeBytes);
        break;
    }
    case hipMemcpyDeviceToDevice: {
        // Need to implement device-to-device copy
        // Option 1: Launch copy kernel
        // Option 2: Copy through host (inefficient)
        // Option 3: DMA if available
        return hipErrorNotSupported;
    }
    case hipMemcpyHostToHost:
        memcpy(dst, src, sizeBytes);
        result = 0;
        break;
    default:
        return hipErrorInvalidValue;
    }

    return (result == 0) ? hipSuccess : hipErrorInvalidValue;
}
```

---

## Kernel Execution

### Kernel Launch

| HIP API | Vortex API | Notes |
|---------|------------|-------|
| `hipLaunchKernel(func, grid, block, args, sharedMem, stream)` | `vx_start(device, kernel, arguments)` | Launch kernel with grid/block configuration |
| Triple-chevron `<<<grid, block>>>` | Compiler transforms to `hipLaunchKernel` | Syntax sugar for kernel launch |

**Implementation Template**:
```cpp
// Kernel registry (populated by compiler-generated registration code)
struct VortexKernelInfo {
    std::string name;
    vx_buffer_h kernel_binary;
    size_t num_args;
};

static std::unordered_map<const void*, VortexKernelInfo> g_kernel_registry;

hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream) {
    if (g_vortex_device == nullptr) return hipErrorInitializationError;

    // Find kernel
    auto it = g_kernel_registry.find(function_address);
    if (it == g_kernel_registry.end()) {
        return hipErrorInvalidDeviceFunction;
    }

    const auto& kernel_info = it->second;

    // Prepare kernel arguments
    // Format: grid_dim (3x uint32), block_dim (3x uint32), then kernel args
    std::vector<uint8_t> arg_buffer;

    // Grid dimensions
    uint32_t grid_dims[3] = {numBlocks.x, numBlocks.y, numBlocks.z};
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(grid_dims),
                     reinterpret_cast<uint8_t*>(grid_dims) + sizeof(grid_dims));

    // Block dimensions
    uint32_t block_dims[3] = {dimBlocks.x, dimBlocks.y, dimBlocks.z};
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(block_dims),
                     reinterpret_cast<uint8_t*>(block_dims) + sizeof(block_dims));

    // Shared memory size
    uint64_t shared_mem = sharedMemBytes;
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(&shared_mem),
                     reinterpret_cast<uint8_t*>(&shared_mem) + sizeof(shared_mem));

    // Marshal kernel arguments
    // This requires metadata about argument types (pointer vs value, size)
    // For now, assume all args are 64-bit values or pointers
    for (size_t i = 0; i < kernel_info.num_args; i++) {
        uint64_t arg_value = *reinterpret_cast<uint64_t*>(args[i]);
        arg_buffer.insert(arg_buffer.end(),
                         reinterpret_cast<uint8_t*>(&arg_value),
                         reinterpret_cast<uint8_t*>(&arg_value) + sizeof(arg_value));
    }

    // Upload arguments to device
    vx_buffer_h arg_buffer_device;
    int result = vx_upload_bytes(g_vortex_device,
                                  arg_buffer.data(),
                                  arg_buffer.size(),
                                  &arg_buffer_device);
    if (result != 0) return hipErrorLaunchFailure;

    // Launch kernel
    result = vx_start(g_vortex_device,
                      kernel_info.kernel_binary,
                      arg_buffer_device);

    // Note: Vortex kernel launch is asynchronous
    // Cleanup of arg_buffer_device should happen after kernel completes
    // For now, leak it or track for later cleanup

    return (result == 0) ? hipSuccess : hipErrorLaunchFailure;
}
```

---

## Kernel API Mappings (Device-Side)

### Thread Indexing

| HIP Built-in | Vortex Built-in | Type | Description |
|--------------|-----------------|------|-------------|
| `threadIdx.x` | `threadIdx.x` | `uint32_t` | **IDENTICAL** - Thread index within block (x dimension) |
| `threadIdx.y` | `threadIdx.y` | `uint32_t` | **IDENTICAL** - Thread index within block (y dimension) |
| `threadIdx.z` | `threadIdx.z` | `uint32_t` | **IDENTICAL** - Thread index within block (z dimension) |
| `blockIdx.x` | `blockIdx.x` | `uint32_t` | **IDENTICAL** - Block index within grid (x dimension) |
| `blockIdx.y` | `blockIdx.y` | `uint32_t` | **IDENTICAL** - Block index within grid (y dimension) |
| `blockIdx.z` | `blockIdx.z` | `uint32_t` | **IDENTICAL** - Block index within grid (z dimension) |
| `blockDim.x` | `blockDim.x` | `uint32_t` | **IDENTICAL** - Block dimensions (x) |
| `blockDim.y` | `blockDim.y` | `uint32_t` | **IDENTICAL** - Block dimensions (y) |
| `blockDim.z` | `blockDim.z` | `uint32_t` | **IDENTICAL** - Block dimensions (z) |
| `gridDim.x` | `gridDim.x` | `uint32_t` | **IDENTICAL** - Grid dimensions (x) |
| `gridDim.y` | `gridDim.y` | `uint32_t` | **IDENTICAL** - Grid dimensions (y) |
| `gridDim.z` | `gridDim.z` | `uint32_t` | **IDENTICAL** - Grid dimensions (z) |

**Key Insight**: Vortex thread indexing is **identical** to HIP/CUDA! No translation needed.

```cpp
// Example: Computing global thread ID (works identically in HIP and Vortex)
__global__ void kernel() {
    // 1D indexing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2D indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 3D indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
}
```

---

## Synchronization Primitives

### Barrier Synchronization

| HIP Function | Vortex Function | Notes |
|--------------|-----------------|-------|
| `__syncthreads()` | `__syncthreads()` | **IDENTICAL** - Block-level barrier |
| `__syncthreads_count(int predicate)` | Custom implementation | Count threads where predicate is true |
| `__syncthreads_and(int predicate)` | Custom implementation | Logical AND across threads |
| `__syncthreads_or(int predicate)` | Custom implementation | Logical OR across threads |

**Vortex Implementation**:
```c
// In vx_spawn.h
#define __syncthreads() vx_barrier(__local_group_id, __warps_per_group)

// Vortex barrier is a hardware instruction (zero overhead)
inline void vx_barrier(int barrier_id, int num_warps) {
    __asm__ volatile (".insn r %0, 4, 0, x0, %1, %2"
                      :: "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(num_warps));
}
```

**Extended Sync Functions** (can be implemented using Vortex primitives):
```cpp
// __syncthreads_count: Returns number of threads where predicate != 0
__device__ int __syncthreads_count(int predicate) {
    // Use warp-level voting + reduction
    int count = 0;
    // Implementation using vx_vote_ballot and bit counting
    __syncthreads();
    return count;
}
```

---

## Warp/Wavefront Operations

### Warp Voting

| HIP Function | Vortex Function | Description |
|--------------|-----------------|-------------|
| `__all(int predicate)` | `vx_vote_all(int predicate)` | True if predicate true for all threads in warp |
| `__any(int predicate)` | `vx_vote_any(int predicate)` | True if predicate true for any thread in warp |
| `__ballot(int predicate)` | `vx_vote_ballot(int predicate)` | Bitmask of threads where predicate is true |

**Mapping**:
```cpp
// HIP → Vortex mapping for warp voting
#define __all(pred)    vx_vote_all(pred)
#define __any(pred)    vx_vote_any(pred)
#define __ballot(pred) vx_vote_ballot(pred)
```

### Warp Shuffle

| HIP Function | Vortex Function | Description |
|--------------|-----------------|-------------|
| `__shfl(int var, int srcLane)` | `vx_shfl_idx(var, srcLane, warpSize, 0xFFFFFFFF)` | Get var from srcLane |
| `__shfl_up(int var, int delta)` | `vx_shfl_up(var, delta, warpSize, 0xFFFFFFFF)` | Get var from lane-delta |
| `__shfl_down(int var, int delta)` | `vx_shfl_down(var, delta, warpSize, 0xFFFFFFFF)` | Get var from lane+delta |
| `__shfl_xor(int var, int laneMask)` | `vx_shfl_bfly(var, laneMask, warpSize, 0xFFFFFFFF)` | Butterfly shuffle |

**Mapping**:
```cpp
// HIP → Vortex shuffle operations
template<typename T>
__device__ T __shfl(T var, int srcLane, int width = warpSize) {
    return vx_shfl_idx(var, srcLane, width, 0xFFFFFFFF);
}

template<typename T>
__device__ T __shfl_up(T var, unsigned int delta, int width = warpSize) {
    return vx_shfl_up(var, delta, width, 0xFFFFFFFF);
}

template<typename T>
__device__ T __shfl_down(T var, unsigned int delta, int width = warpSize) {
    return vx_shfl_down(var, delta, width, 0xFFFFFFFF);
}

template<typename T>
__device__ T __shfl_xor(T var, int laneMask, int width = warpSize) {
    return vx_shfl_bfly(var, laneMask, width, 0xFFFFFFFF);
}
```

---

## Shared Memory

### Allocation

| HIP Syntax | Vortex Syntax | Notes |
|------------|---------------|-------|
| `__shared__ float data[256]` | `__shared__ float* data = __local_mem(256 * sizeof(float))` | Static vs dynamic allocation |
| `extern __shared__ float data[]` | `__shared__ float* data = __local_mem(extern_shared_mem_size)` | Dynamic shared memory |

**Vortex Shared Memory**:
```c
// In vx_spawn.h
#define __local_mem(size) \
    (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)
```

**HIP-Compatible Wrapper**:
```cpp
// For static shared memory
#define __shared__ __attribute__((address_space(3)))

// For dynamic shared memory in HIP:
extern __shared__ float shared_mem[];

// In Vortex, get dynamic shared memory size from kernel args
// and use __local_mem()
```

---

## Atomic Operations

### Basic Atomics

| HIP Function | Vortex Implementation | Notes |
|--------------|----------------------|-------|
| `atomicAdd(int* addr, int val)` | RISC-V `amoadd.w` instruction | Use RISC-V atomic extension |
| `atomicSub(int* addr, int val)` | `amoadd.w` with negated val | Subtract via atomic add |
| `atomicExch(int* addr, int val)` | `amoswap.w` instruction | Atomic exchange |
| `atomicMin(int* addr, int val)` | `amomin.w` instruction | Atomic minimum |
| `atomicMax(int* addr, int val)` | `amomax.w` instruction | Atomic maximum |
| `atomicCAS(int* addr, int cmp, int val)` | Custom using `lr.w`/`sc.w` | Compare-and-swap |

**Implementation Template**:
```cpp
__device__ int atomicAdd(int* address, int val) {
    int old;
    asm volatile(
        "amoadd.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

__device__ int atomicSub(int* address, int val) {
    return atomicAdd(address, -val);
}

__device__ int atomicExch(int* address, int val) {
    int old;
    asm volatile(
        "amoswap.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

__device__ int atomicMin(int* address, int val) {
    int old;
    asm volatile(
        "amomin.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

__device__ int atomicMax(int* address, int val) {
    int old;
    asm volatile(
        "amomax.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}

__device__ int atomicCAS(int* address, int compare, int val) {
    int old;
    asm volatile(
        "1: lr.w %0, (%1)\n"
        "   bne %0, %2, 2f\n"
        "   sc.w t0, %3, (%1)\n"
        "   bnez t0, 1b\n"
        "2:"
        : "=&r"(old)
        : "r"(address), "r"(compare), "r"(val)
        : "t0", "memory"
    );
    return old;
}
```

---

## Math Functions

Most HIP math functions map directly to standard C math library or RISC-V floating-point instructions:

| HIP Function | Vortex Implementation | Notes |
|--------------|----------------------|-------|
| `__fsqrt_rn(float x)` | `fsqrt.s` instruction | Hardware square root |
| `__fmul_rn(float a, float b)` | `fmul.s` instruction | Hardware multiply |
| `__fadd_rn(float a, float b)` | `fadd.s` instruction | Hardware add |
| `__fmaf(float a, float b, float c)` | `fmadd.s` instruction | Fused multiply-add |
| `sinf(float x)` | Software implementation | Use libm or approximation |
| `cosf(float x)` | Software implementation | Use libm or approximation |
| `expf(float x)` | Software implementation | Use libm or approximation |

---

## Complete Kernel Example

### HIP Kernel
```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Vortex Kernel (Identical!)
```cpp
// Kernel argument structure
struct kernel_arg_t {
    float* a;
    float* b;
    float* c;
    int n;
};

void vectorAdd_kernel(kernel_arg_t* arg) {
    // Thread indexing works identically!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < arg->n) {
        arg->c[i] = arg->a[i] + arg->b[i];
    }
}

// Entry point for Vortex kernel
int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    uint32_t grid_dim[3];   // Passed in args
    uint32_t block_dim[3];  // Passed in args

    return vx_spawn_threads(3, grid_dim, block_dim,
                           (vx_kernel_func_cb)vectorAdd_kernel, arg);
}
```

---

## Summary Table: API Coverage

### Runtime API (Host-Side)

| Category | HIP Functions | Direct Vortex Mapping | Implementation Effort |
|----------|---------------|----------------------|----------------------|
| Device Management | 4 | ✅ Yes | Low (1-2 days) |
| Memory Allocation | 4 | ✅ Yes | Low (1-2 days) |
| Memory Transfer | 3 | ✅ Yes | Low (1-2 days) |
| Kernel Launch | 1 | ⚠️ Partial | Medium (3-5 days) |
| Synchronization | 1 | ✅ Yes | Low (1 day) |

**Total Runtime API Implementation**: ~1-2 weeks

### Kernel API (Device-Side)

| Category | HIP Functions | Direct Vortex Mapping | Implementation Effort |
|----------|---------------|----------------------|----------------------|
| Thread Indexing | 12 | ✅ **Identical!** | None (0 days) |
| Synchronization | 1 | ✅ **Identical!** | None (0 days) |
| Warp Voting | 3 | ✅ Direct mapping | Low (1 day) |
| Warp Shuffle | 4 | ✅ Direct mapping | Low (1 day) |
| Shared Memory | 1 | ⚠️ Syntax difference | Low (1 day) |
| Atomics | 6+ | ✅ RISC-V instructions | Medium (2-3 days) |
| Math Functions | 50+ | ⚠️ Mixed | Medium (varies) |

**Total Kernel API Implementation**: ~1-2 weeks

---

## Implementation Strategy

### Phase 1: Minimal Runtime (Week 1)
Implement these 15 functions:
1. `hipInit`
2. `hipGetDeviceCount`
3. `hipSetDevice`
4. `hipGetDevice`
5. `hipGetDeviceProperties`
6. `hipMalloc`
7. `hipFree`
8. `hipMemcpy` (H2D, D2H)
9. `hipMemset`
10. `hipLaunchKernel` (basic)
11. `hipDeviceSynchronize`
12. `hipGetLastError`
13. `hipGetErrorString`
14. `hipMallocHost`
15. `hipFreeHost`

**Test**: Simple vector addition

### Phase 2: Kernel API (Week 2)
Device-side implementations:
- Thread indexing (already works!)
- `__syncthreads()` (already works!)
- Warp voting (mapping)
- Warp shuffles (mapping)
- Basic atomics

**Test**: Matrix multiplication with shared memory

### Phase 3: Advanced Features (Week 3-4)
- Streams and events
- Async memory operations
- Extended atomics
- Math library functions

**Test**: Full HIP test suite

---

## Reference Implementation: Vector Addition

### Complete HIP Program
```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    printf("c[0] = %f (expected 0)\n", h_c[0]);
    printf("c[1] = %f (expected 3)\n", h_c[1]);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```

### Vortex API Calls (Behind the Scenes)
```cpp
// hipMalloc → vx_mem_alloc
vx_mem_alloc(device, size, VX_MEM_READ_WRITE, &buffer_a);
vx_mem_alloc(device, size, VX_MEM_READ_WRITE, &buffer_b);
vx_mem_alloc(device, size, VX_MEM_READ_WRITE, &buffer_c);

// hipMemcpy H2D → vx_copy_to_dev
vx_copy_to_dev(buffer_a, h_a, 0, size);
vx_copy_to_dev(buffer_b, h_b, 0, size);

// vectorAdd<<<...>>> → vx_start
vx_upload_kernel_file(device, "vectorAdd.vxbin", &kernel);
vx_upload_bytes(device, &args, sizeof(args), &arg_buffer);
vx_start(device, kernel, arg_buffer);

// hipMemcpy D2H → vx_copy_from_dev
vx_copy_from_dev(h_c, buffer_c, 0, size);

// hipDeviceSynchronize → vx_ready_wait
vx_ready_wait(device, VX_MAX_TIMEOUT);

// hipFree → vx_mem_free
vx_mem_free(buffer_a);
vx_mem_free(buffer_b);
vx_mem_free(buffer_c);
```

---

## Conclusion

**Key Findings**:

1. ✅ **Thread indexing is identical** - No translation needed
2. ✅ **Barriers are identical** - Hardware support in both
3. ✅ **Memory model maps directly** - Clear correspondence
4. ✅ **Warp operations have direct equivalents** - Simple mapping
5. ⚠️ **Kernel launch requires wrapper** - Medium complexity

**Estimated Implementation Time**:
- **Tier 1 (OpenCL)**: 1 week (just configuration)
- **Tier 2 (Extensions)**: 2-3 weeks total
- **Tier 3 (Direct backend)**: 2-3 months

**Recommendation**: Start with Tier 1 (use chipStar + OpenCL), then add Tier 2 extensions for Vortex-specific optimizations.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-05
**Status**: Complete API mapping
**Next Step**: Implement runtime wrapper using these mappings
