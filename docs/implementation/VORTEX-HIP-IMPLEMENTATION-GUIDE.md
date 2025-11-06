# Vortex HIP Implementation Guide

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Decision Matrix](#architecture-decision-matrix)
3. [Minimal Runtime Implementation](#minimal-runtime-implementation)
4. [Compilation Pipeline Options](#compilation-pipeline-options)
5. [Memory Management Design](#memory-management-design)
6. [Kernel Launch Mechanism](#kernel-launch-mechanism)
7. [Synchronization Primitives](#synchronization-primitives)
8. [Implementation Phases](#implementation-phases)
9. [Testing Strategy](#testing-strategy)
10. [Performance Considerations](#performance-considerations)

---

## Executive Summary

This guide provides a concrete roadmap for implementing HIP support on the Vortex RISC-V GPU, based on analysis of two reference implementations:

- **chipStar**: Full-featured SPIR-V based runtime (~50K LOC)
- **HIP-CPU**: Minimal CPU implementation (~2K LOC)

### Recommended Approach

**Hybrid Strategy:**
1. Start with HIP-CPU-inspired minimal runtime (quick validation)
2. Adopt chipStar's compilation pipeline (SPIR-V based)
3. Implement custom Vortex backend (similar to OpenCL backend)
4. Incrementally add features

**Estimated Complexity:** ~10,000-15,000 lines of C++ code

---

## Architecture Decision Matrix

### Decision 1: Intermediate Representation

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **SPIR-V** | Standard IR, tooling, reuse chipStar passes | Requires SPIR-V support, translation overhead | ✅ **Recommended** if Vortex has/can add SPIR-V support |
| **LLVM IR** | Direct RISC-V codegen, no translation | Custom passes needed, less portable | Consider if SPIR-V infeasible |
| **Custom Binary** | Optimal for Vortex | Most implementation work, no ecosystem | Avoid unless necessary |

**Recommendation:** Use SPIR-V if possible, falling back to LLVM IR if not.

### Decision 2: Memory Model

| Model | Implementation | Pros | Cons |
|-------|---------------|------|------|
| **Unified Addressing** | Host ptr = Device ptr | Simple, HIP-CPU approach | Requires hardware support |
| **Separate Address Spaces** | Host ptr ≠ Device ptr | Standard GPU model, chipStar approach | More complex runtime |
| **Hybrid** | Default unified, explicit device alloc available | Flexibility | Complexity |

**Recommendation:** Start with unified if hardware supports it, implement separate spaces if needed.

### Decision 3: Compilation Strategy

| Strategy | When | Tradeoff |
|----------|------|----------|
| **JIT (Just-In-Time)** | Runtime (chipStar) | Flexibility, slower first launch |
| **AOT (Ahead-Of-Time)** | Build time | Fast launch, no runtime compiler |
| **Hybrid** | AOT with JIT fallback | Best of both | Complex |

**Recommendation:** Start with JIT (easier debugging), add AOT optimization later.

---

## Minimal Runtime Implementation

### Phase 1: Core API (Week 1-2)

Implement the absolute minimum to run a simple kernel:

```cpp
// File: vortex_hip_runtime.h

#ifndef VORTEX_HIP_RUNTIME_H
#define VORTEX_HIP_RUNTIME_H

#include <hip/hip_runtime_api.h>

// Device Management (3 functions)
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int device);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);

// Memory Management (5 functions)
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
hipError_t hipMemset(void* ptr, int value, size_t size);

// Kernel Execution (2 functions)
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream);
hipError_t hipDeviceSynchronize();

// Error Handling (2 functions)
const char* hipGetErrorString(hipError_t error);
hipError_t hipGetLastError();

#endif
```

### Implementation Template

```cpp
// File: vortex_hip_runtime.cpp

#include "vortex_hip_runtime.h"
#include <vortex_driver.h>  // Your Vortex driver interface

namespace {
    // Global state
    struct VortexDevice {
        vortex_device_h handle;
        bool initialized = false;
    };

    VortexDevice g_device;
    hipError_t g_last_error = hipSuccess;
}

// Device Management
hipError_t hipGetDeviceCount(int* count) {
    if (!count) return hipErrorInvalidValue;

    // Query Vortex driver for device count
    int num_devices = vortex_get_num_devices();
    *count = num_devices;

    return hipSuccess;
}

hipError_t hipSetDevice(int device) {
    if (device < 0) return hipErrorInvalidDevice;

    // Initialize Vortex device if not already done
    if (!g_device.initialized) {
        int result = vortex_dev_open(&g_device.handle);
        if (result != 0) return hipErrorInitializationError;
        g_device.initialized = true;
    }

    return hipSuccess;
}

// Memory Management
hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) return hipErrorInvalidValue;
    if (size == 0) return hipSuccess;

    // Allocate on Vortex device
    uint64_t dev_addr;
    int result = vortex_mem_alloc(g_device.handle, size, &dev_addr);
    if (result != 0) return hipErrorOutOfMemory;

    *ptr = reinterpret_cast<void*>(dev_addr);
    return hipSuccess;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) return hipSuccess;

    uint64_t dev_addr = reinterpret_cast<uint64_t>(ptr);
    int result = vortex_mem_free(g_device.handle, dev_addr);
    if (result != 0) return hipErrorInvalidValue;

    return hipSuccess;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (!dst || !src) return hipErrorInvalidValue;
    if (size == 0) return hipSuccess;

    int result;
    switch (kind) {
    case hipMemcpyHostToDevice:
        result = vortex_copy_to_dev(g_device.handle,
                                     reinterpret_cast<uint64_t>(dst),
                                     src, size);
        break;
    case hipMemcpyDeviceToHost:
        result = vortex_copy_from_dev(g_device.handle,
                                       dst,
                                       reinterpret_cast<uint64_t>(src),
                                       size);
        break;
    case hipMemcpyDeviceToDevice:
        // May need DMA or explicit copy
        result = vortex_mem_copy_dev(g_device.handle,
                                      reinterpret_cast<uint64_t>(dst),
                                      reinterpret_cast<uint64_t>(src),
                                      size);
        break;
    case hipMemcpyHostToHost:
        memcpy(dst, src, size);
        result = 0;
        break;
    default:
        return hipErrorInvalidValue;
    }

    return (result == 0) ? hipSuccess : hipErrorInvalidValue;
}

// Kernel Execution
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream) {
    // This is the complex part - see Kernel Launch section below

    // 1. Find kernel by function pointer
    // 2. Marshal arguments
    // 3. Submit to Vortex
    // 4. Return immediately (async)

    return hipSuccess;  // Placeholder
}

hipError_t hipDeviceSynchronize() {
    int result = vortex_wait(g_device.handle);
    return (result == 0) ? hipSuccess : hipErrorUnknown;
}
```

### Test Program

```cpp
// test_minimal.cpp
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

    // Host arrays
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // Copy to device
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_c, N);

    // Copy result back
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }

    printf("%s\n", success ? "PASSED" : "FAILED");

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return success ? 0 : 1;
}
```

---

## Compilation Pipeline Options

### Option A: Adapt chipStar Pipeline (Recommended)

```
HIP Source (.hip, .cu)
      ↓
Clang Frontend (hip-clang wrapper)
  Flags: --offload=spirv64 -target spirv64
      ↓
LLVM IR (device code only)
      ↓
Link chipStar device library
  (builtins, math functions)
      ↓
chipStar LLVM Passes:
  - HipTextureLowering
  - HipKernelArgSpiller
  - HipDynMem
  - HipGlobalVariables
      ↓
SPIRV-LLVM-Translator
      ↓
SPIR-V Binary
      ↓
[NEW] SPIR-V → Vortex ISA Translator
      ↓
Vortex Binary
      ↓
Embed in fat binary (.hip_fatbin section)
      ↓
Link with Vortex runtime
      ↓
Executable
```

### Option B: Direct LLVM → RISC-V

```
HIP Source
      ↓
Clang Frontend
      ↓
LLVM IR
      ↓
Custom Vortex LLVM Passes
      ↓
LLVM RISC-V Backend + Vortex Extensions
      ↓
RISC-V Assembly with Vortex GPU instructions
      ↓
Vortex Assembler
      ↓
Vortex Binary
      ↓
Embed in fat binary
```

### Compilation Script

```bash
#!/bin/bash
# vortex-hipcc wrapper

SOURCE=$1
OUTPUT=$2

# Environment variables (configure for your system)
VORTEX_ROOT=${VORTEX_ROOT}
HIP_INSTALL=${HIP_INSTALL}

# Step 1: Compile device code to SPIR-V
clang++ \
    -x hip \
    --cuda-device-only \
    --offload=spirv64 \
    -target spirv64 \
    --hip-path=/path/to/hip \
    -Xclang -load -Xclang libchipStar.so \
    -Xclang -add-plugin -Xclang hip-lower \
    -c $SOURCE \
    -o ${SOURCE}.bc

# Step 2: Run LLVM passes
opt \
    -load=libchipStarPasses.so \
    -hip-post-process-spir-v \
    ${SOURCE}.bc \
    -o ${SOURCE}.opt.bc

# Step 3: Translate to SPIR-V
llvm-spirv ${SOURCE}.opt.bc -o ${SOURCE}.spv

# Step 4: [NEW] Translate SPIR-V to Vortex
vortex-spirv-translator ${SOURCE}.spv -o ${SOURCE}.vbin

# Step 5: Embed in fat binary
clang-offload-bundler \
    -type=o \
    -targets=host-x86_64,vortex \
    -inputs=${SOURCE}.host.o,${SOURCE}.vbin \
    -outputs=${OUTPUT}

# Step 6: Link with BOTH runtime libraries
# CRITICAL: HIP programs must link to:
#   1. Vortex runtime library (for GPU driver/hardware access)
#   2. HIP runtime library (for HIP API implementation)
clang++ \
    ${OUTPUT} \
    -L${VORTEX_ROOT}/stub \
    -lvortex \
    -L${HIP_INSTALL}/lib \
    -lCHIP \
    -o final_executable
```

---

## Memory Management Design

### Option 1: Unified Virtual Memory (Simplest)

```cpp
struct VortexMemoryManager {
    // Unified address space - all pointers valid everywhere

    hipError_t allocate(void** ptr, size_t size) {
        // Allocate in shared CPU-GPU address space
        *ptr = vortex_unified_alloc(size);
        return hipSuccess;
    }

    hipError_t copy(void* dst, void* src, size_t size, hipMemcpyKind kind) {
        // All copies are just memcpy (or DMA for perf)
        if (kind == hipMemcpyDeviceToDevice) {
            // Use DMA if available
            vortex_dma_copy(dst, src, size);
        } else {
            memcpy(dst, src, size);
        }
        return hipSuccess;
    }
};
```

### Option 2: Separate Address Spaces (More Realistic)

```cpp
struct VortexMemoryManager {
    struct Allocation {
        void* host_ptr;
        uint64_t device_addr;
        size_t size;
    };

    std::unordered_map<void*, Allocation> allocations;

    hipError_t allocate(void** ptr, size_t size) {
        // Allocate on device
        uint64_t dev_addr;
        vortex_mem_alloc(device, size, &dev_addr);

        // Create host-side tracking pointer
        void* host_ptr = reinterpret_cast<void*>(dev_addr);

        allocations[host_ptr] = {host_ptr, dev_addr, size};
        *ptr = host_ptr;

        return hipSuccess;
    }

    hipError_t copy(void* dst, void* src, size_t size, hipMemcpyKind kind) {
        switch (kind) {
        case hipMemcpyHostToDevice: {
            auto it = allocations.find(dst);
            if (it == allocations.end()) return hipErrorInvalidValue;
            vortex_copy_to_dev(device, it->second.device_addr, src, size);
            break;
        }
        case hipMemcpyDeviceToHost: {
            auto it = allocations.find(const_cast<void*>(src));
            if (it == allocations.end()) return hipErrorInvalidValue;
            vortex_copy_from_dev(device, dst, it->second.device_addr, size);
            break;
        }
        // ... other cases
        }
        return hipSuccess;
    }
};
```

---

## Kernel Launch Mechanism

### Key Components

1. **Kernel Registry** (adapted from SPVRegister)
2. **Argument Marshalling** (from chipStar)
3. **Launch Submission** (Vortex-specific)

### Implementation

```cpp
// Kernel Registry
class VortexKernelRegistry {
    struct KernelInfo {
        std::string name;
        void* device_function;  // Pointer to device code
        std::vector<ArgType> arg_types;
        size_t shared_mem_size;
    };

    std::unordered_map<const void*, KernelInfo> kernels;

public:
    void registerKernel(const void* host_ptr, const KernelInfo& info) {
        kernels[host_ptr] = info;
    }

    const KernelInfo* findKernel(const void* host_ptr) {
        auto it = kernels.find(host_ptr);
        return (it != kernels.end()) ? &it->second : nullptr;
    }
};

// Kernel Launch
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream) {
    // 1. Find kernel
    auto* kernel_info = g_registry.findKernel(func);
    if (!kernel_info) return hipErrorInvalidDeviceFunction;

    // 2. Allocate argument buffer
    std::vector<uint8_t> arg_buffer;
    for (size_t i = 0; i < kernel_info->arg_types.size(); i++) {
        const auto& arg_type = kernel_info->arg_types[i];
        void* arg_data = args[i];

        if (arg_type.is_pointer) {
            // Pass device address
            uint64_t dev_addr = reinterpret_cast<uint64_t>(arg_data);
            arg_buffer.insert(arg_buffer.end(),
                             reinterpret_cast<uint8_t*>(&dev_addr),
                             reinterpret_cast<uint8_t*>(&dev_addr) + sizeof(dev_addr));
        } else {
            // Pass value directly
            arg_buffer.insert(arg_buffer.end(),
                             reinterpret_cast<uint8_t*>(arg_data),
                             reinterpret_cast<uint8_t*>(arg_data) + arg_type.size);
        }
    }

    // 3. Setup Vortex kernel launch
    vortex_kernel_config_t config;
    config.grid_dim[0] = grid.x;
    config.grid_dim[1] = grid.y;
    config.grid_dim[2] = grid.z;
    config.block_dim[0] = block.x;
    config.block_dim[1] = block.y;
    config.block_dim[2] = block.z;
    config.shared_mem_size = sharedMem + kernel_info->shared_mem_size;

    // 4. Submit kernel
    int result = vortex_kernel_launch(
        g_device.handle,
        kernel_info->device_function,
        &config,
        arg_buffer.data(),
        arg_buffer.size()
    );

    return (result == 0) ? hipSuccess : hipErrorLaunchFailure;
}
```

---

## Synchronization Primitives

### Hardware Barriers (Preferred)

If Vortex has hardware barrier support:

```cpp
__device__ void __syncthreads() {
    // Maps to Vortex barrier instruction
    asm volatile("vortex.barrier" ::: "memory");
}
```

### Shared Memory

```cpp
// In device code:
__shared__ float shared_data[256];

// Runtime ensures each block gets its own copy
// Vortex needs to provide per-block memory regions
```

### Atomics

```cpp
__device__ int atomicAdd(int* address, int val) {
    // Maps to RISC-V atomic instruction
    int old;
    asm volatile(
        "amoadd.w %0, %2, (%1)"
        : "=r"(old)
        : "r"(address), "r"(val)
        : "memory"
    );
    return old;
}
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 weeks)

**Deliverables:**
- [ ] Basic runtime library (hipMalloc, hipMemcpy, hipFree)
- [ ] Device initialization
- [ ] Simple kernel launch (no arguments)
- [ ] Test: Launch empty kernel, synchronize

**Success Criteria:**
```cpp
__global__ void empty_kernel() { }

int main() {
    empty_kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    return 0;
}
```

### Phase 2: Basic Kernels (2-3 weeks)

**Deliverables:**
- [ ] Argument marshalling
- [ ] Built-in variables (threadIdx, blockIdx, blockDim, gridDim)
- [ ] Memory operations in kernels

**Success Criteria:**
- Vector addition test passes
- Matrix multiplication (no shared memory)

### Phase 3: Advanced Features (3-4 weeks)

**Deliverables:**
- [ ] Shared memory support
- [ ] __syncthreads() barriers
- [ ] Atomic operations
- [ ] Stream support

**Success Criteria:**
- Parallel reduction works
- Matrix multiplication with shared memory

### Phase 4: Optimization (2-3 weeks)

**Deliverables:**
- [ ] Kernel caching
- [ ] Async operations
- [ ] DMA optimization
- [ ] Error handling

### Phase 5: Compatibility (Ongoing)

**Deliverables:**
- [ ] Pass HIP conformance tests
- [ ] Performance benchmarks
- [ ] Documentation

---

## Testing Strategy

### Unit Tests

```cpp
// test_memory.cpp
TEST(VortexHIP, BasicAllocation) {
    void* ptr = nullptr;
    ASSERT_EQ(hipMalloc(&ptr, 1024), hipSuccess);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(hipFree(ptr), hipSuccess);
}

TEST(VortexHIP, MemcpyHostToDevice) {
    const size_t size = 1024;
    float* h_data = new float[size];
    float* d_data = nullptr;

    for (size_t i = 0; i < size; i++) h_data[i] = i;

    ASSERT_EQ(hipMalloc(&d_data, size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_data, h_data, size * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);

    float* h_result = new float[size];
    ASSERT_EQ(hipMemcpy(h_result, d_data, size * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);

    for (size_t i = 0; i < size; i++) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }

    delete[] h_data;
    delete[] h_result;
    hipFree(d_data);
}
```

### Integration Tests

```cpp
// test_kernels.cpp
__global__ void add_kernel(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

TEST(VortexHIP, SimpleKernel) {
    const int N = 1024;
    // ... allocate and copy ...

    add_kernel<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);

    // ... copy back and verify ...
}
```

### Performance Benchmarks

```cpp
// benchmark_bandwidth.cpp
void benchmark_memcpy_bandwidth() {
    const size_t sizes[] = {1<<10, 1<<20, 1<<30};  // 1KB, 1MB, 1GB

    for (size_t size : sizes) {
        void* d_ptr;
        hipMalloc(&d_ptr, size);
        void* h_ptr = malloc(size);

        auto start = high_resolution_clock::now();
        hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        auto end = high_resolution_clock::now();

        double seconds = duration_cast<microseconds>(end - start).count() / 1e6;
        double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / seconds;  // GB/s

        printf("Size: %zu bytes, Bandwidth: %.2f GB/s\n", size, bandwidth);

        hipFree(d_ptr);
        free(h_ptr);
    }
}
```

---

## Performance Considerations

### Critical Performance Factors

1. **Kernel Launch Overhead**
   - Target: < 10 μs per launch
   - Optimize argument marshalling
   - Cache compiled kernels

2. **Memory Bandwidth**
   - Measure with bandwidth benchmark
   - Optimize DMA if available
   - Consider async copies

3. **Barrier Overhead**
   - Must be hardware-supported
   - HIP-CPU shows software barriers are 10-100x slower

4. **Thread Scalability**
   - How many threads can Vortex run simultaneously?
   - Optimize block size recommendations

### Optimization Checklist

- [ ] Kernel caching (avoid recompilation)
- [ ] Argument buffer pooling
- [ ] Async memory copies
- [ ] Zero-copy for unified memory
- [ ] DMA for large transfers
- [ ] Batch small operations

---

## Summary

### Recommended Implementation Path

1. **Week 1-2:** Minimal runtime (memory + empty kernels)
2. **Week 3-4:** Basic kernels with arguments
3. **Week 5-6:** Shared memory + barriers
4. **Week 7-8:** Streams + async operations
5. **Week 9-10:** Optimization + testing
6. **Week 11-12:** Compatibility + documentation

**Total Time Estimate:** 3 months for basic functionality

### Key Success Factors

1. ✅ Hardware barrier support (critical)
2. ✅ Efficient memory transfers
3. ✅ SPIR-V or LLVM IR compilation path
4. ✅ Clear driver interface specification
5. ✅ Comprehensive testing

### Risk Mitigation

- Start with HIP-CPU approach for quick validation
- Adopt chipStar patterns incrementally
- Build comprehensive test suite early
- Measure performance continuously

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Target Platform:** Vortex RISC-V GPU
**Status:** Implementation Planning
