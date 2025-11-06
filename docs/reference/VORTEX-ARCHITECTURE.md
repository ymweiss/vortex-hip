# Vortex GPU Architecture for HIP Implementation

## Executive Summary

Vortex is an open-source RISC-V based GPGPU that **already supports OpenCL 1.2** via POCL. This means **most of the infrastructure needed for HIP is already in place**. The path to HIP support is straightforward: adapt chipStar's approach to use Vortex as a backend, similar to how chipStar uses OpenCL.

**Key Insight:** Vortex + POCL + chipStar = Complete HIP Solution

---

## Table of Contents

1. [Vortex Overview](#vortex-overview)
2. [Runtime API](#runtime-api)
3. [Kernel API](#kernel-api)
4. [Existing OpenCL Support](#existing-opencl-support)
5. [Path to HIP Implementation](#path-to-hip-implementation)
6. [Hardware Capabilities](#hardware-capabilities)

---

## Vortex Overview

### What is Vortex?

**Vortex** is a full-stack open-source RISC-V GPGPU with:
- **RISC-V ISA**: RV32IMAF and RV64IMAFD support
- **Warp-based execution**: Similar to NVIDIA GPUs
- **OpenCL 1.2 support**: Via POCL (Portable OpenCL)
- **Multiple backends**: SimX (C++ simulator), RTL simulator, FPGA
- **Configurable**: Cores, warps, threads, caches

### Key Specifications

| Feature | Details |
|---------|---------|
| ISA | RISC-V RV32IMAF, RV64IMAFD |
| Software | OpenCL 1.2 |
| Execution Model | Warp-based (SIMT) |
| Cores | Configurable |
| Warps per Core | Configurable |
| Threads per Warp | Configurable |
| Memory | L1, L2, L3 caches, local memory |
| Backends | SimX, RTL simulator, Altera/Xilinx FPGAs |

### Why Vortex is Perfect for HIP

✅ **Already has warp-based execution** (like CUDA/HIP)
✅ **Already has OpenCL support** (chipStar targets OpenCL)
✅ **Already has __syncthreads()** (vx_barrier)
✅ **Already has thread indexing** (threadIdx, blockIdx)
✅ **Already has shared memory** (__local_mem)
✅ **RISC-V based** (open, extensible)

---

## Runtime API

### Host-Side API (vortex.h)

The Vortex runtime API for host code:

```c
// Device Management
int vx_dev_open(vx_device_h* hdevice);
int vx_dev_close(vx_device_h hdevice);
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value);

// Device Capabilities
#define VX_CAPS_VERSION             0x0
#define VX_CAPS_NUM_THREADS         0x1  // Threads per warp
#define VX_CAPS_NUM_WARPS           0x2  // Warps per core
#define VX_CAPS_NUM_CORES           0x3  // Number of cores
#define VX_CAPS_CACHE_LINE_SIZE     0x4
#define VX_CAPS_GLOBAL_MEM_SIZE     0x5
#define VX_CAPS_LOCAL_MEM_SIZE      0x6  // Shared memory size
#define VX_CAPS_ISA_FLAGS           0x7

// Memory Management
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);
int vx_mem_free(vx_buffer_h hbuffer);
int vx_mem_address(vx_buffer_h hbuffer, uint64_t* address);
int vx_mem_info(vx_device_h hdevice, uint64_t* mem_free, uint64_t* mem_used);

// Memory Transfer
int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr,
                   uint64_t dst_offset, uint64_t size);
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer,
                     uint64_t src_offset, uint64_t size);

// Kernel Execution
int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h harguments);
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

// Utility
int vx_upload_kernel_file(vx_device_h hdevice, const char* filename,
                          vx_buffer_h* hbuffer);
int vx_upload_bytes(vx_device_h hdevice, const void* content, uint64_t size,
                   vx_buffer_h* hbuffer);
```

### Memory Flags

```c
#define VX_MEM_READ                 0x1
#define VX_MEM_WRITE                0x2
#define VX_MEM_READ_WRITE           0x3
#define VX_MEM_PIN_MEMORY           0x4
```

### Example Usage

```cpp
#include <vortex.h>

int main() {
    vx_device_h device;

    // 1. Open device
    vx_dev_open(&device);

    // 2. Query capabilities
    uint64_t num_cores, num_warps, num_threads;
    vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores);
    vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps);
    vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads);

    printf("Device: %lu cores, %lu warps, %lu threads\n",
           num_cores, num_warps, num_threads);

    // 3. Allocate memory
    vx_buffer_h buffer;
    vx_mem_alloc(device, 1024, VX_MEM_READ_WRITE, &buffer);

    // 4. Transfer data
    int data[256];
    vx_copy_to_dev(buffer, data, 0, sizeof(data));

    // 5. Load and run kernel
    vx_buffer_h kernel;
    vx_upload_kernel_file(device, "kernel.vxbin", &kernel);

    vx_buffer_h args;
    vx_upload_bytes(device, &buffer, sizeof(buffer), &args);

    vx_start(device, kernel, args);
    vx_ready_wait(device, VX_MAX_TIMEOUT);

    // 6. Read results
    vx_copy_from_dev(data, buffer, 0, sizeof(data));

    // 7. Cleanup
    vx_mem_free(buffer);
    vx_mem_free(kernel);
    vx_mem_free(args);
    vx_dev_close(device);

    return 0;
}
```

---

## Kernel API

### Device-Side API (vx_spawn.h, vx_intrinsics.h)

The Vortex kernel API for device code:

```c
// Thread Indexing (CUDA/HIP compatible!)
typedef union {
    struct { uint32_t x, y, z; };
    uint32_t m[3];
} dim3_t;

extern __thread dim3_t blockIdx;   // Block index
extern __thread dim3_t threadIdx;  // Thread index within block
extern dim3_t gridDim;             // Grid dimensions
extern dim3_t blockDim;            // Block dimensions

// Synchronization (CUDA/HIP compatible!)
#define __syncthreads() \
    vx_barrier(__local_group_id, __warps_per_group)

// Shared Memory (CUDA/HIP compatible!)
#define __local_mem(size) \
    (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)

// Kernel Launch
int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t* block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg);

// Thread Intrinsics
int vx_thread_id();      // Thread ID within warp
int vx_warp_id();        // Warp ID within core
int vx_core_id();        // Core ID
int vx_hart_id();        // Global hardware thread ID
int vx_num_threads();    // Threads per warp
int vx_num_warps();      // Warps per core
int vx_num_cores();      // Total cores
```

### Thread Control

```c
// Thread masking
void vx_tmc(int thread_mask);     // Set active threads
void vx_tmc_zero();                // Disable all threads
void vx_tmc_one();                 // Enable only thread 0

// Warp operations
void vx_wspawn(int num_warps, vx_wspawn_pfn func_ptr);

// Warp voting
int vx_vote_all(int predicate);    // All threads true?
int vx_vote_any(int predicate);    // Any thread true?
int vx_vote_ballot(int predicate); // Bitmask of threads

// Warp shuffle
int vx_shfl_up(size_t value, int delta, int width, int mask);
int vx_shfl_down(size_t value, int delta, int width, int mask);
int vx_shfl_bfly(size_t value, int mask, int width, int mask);
int vx_shfl_idx(size_t value, int idx, int width, int mask);
```

### Example Kernel

```cpp
#include <vx_spawn.h>

// Kernel argument structure
struct kernel_arg_t {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t num_points;
};

// Kernel body (like CUDA __global__ function)
void vector_add_kernel(kernel_arg_t* arg) {
    float* src = (float*)arg->src_addr;
    float* dst = (float*)arg->dst_addr;

    // Get thread index (CUDA-style)
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < arg->num_points) {
        dst[tid] = src[tid] * 2.0f;
    }
}

// Entry point
int main() {
    // Get kernel arguments from register
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    // Launch threads (like CUDA <<<grid, block>>>)
    uint32_t grid_dim = (arg->num_points + 255) / 256;
    uint32_t block_dim = 256;

    return vx_spawn_threads(1, &grid_dim, &block_dim,
                           (vx_kernel_func_cb)vector_add_kernel, arg);
}
```

### Shared Memory Example

```cpp
void matmul_kernel(kernel_arg_t* arg) {
    // Allocate shared memory (like CUDA __shared__)
    __shared__ float* As = __local_mem(256 * sizeof(float));
    __shared__ float* Bs = __local_mem(256 * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load data into shared memory
    As[ty * 16 + tx] = A[...];
    Bs[ty * 16 + tx] = B[...];

    // Synchronize (like CUDA __syncthreads())
    __syncthreads();

    // Compute using shared memory
    float sum = 0;
    for (int k = 0; k < 16; k++) {
        sum += As[ty * 16 + k] * Bs[k * 16 + tx];
    }

    C[...] = sum;
}
```

---

## Existing OpenCL Support

### POCL Integration

Vortex **already supports OpenCL 1.2** through POCL (Portable OpenCL):

```bash
# OpenCL tests already exist
ls ~/vortex/tests/opencl/

# Examples:
vecadd/        # Vector addition
sgemm/         # Matrix multiplication
saxpy/         # SAXPY operation
dotproduct/    # Dot product
transpose/     # Matrix transpose
```

### OpenCL Kernel Example

```c
// From tests/opencl/saxpy/kernel.cl
__kernel void saxpy(__global float* x,
                    __global float* y,
                    float a,
                    unsigned size) {
    int id = get_global_id(0);
    if (id < size) {
        y[id] = a * x[id] + y[id];
    }
}
```

### How POCL Works with Vortex

```
OpenCL Application
        ↓
POCL Runtime (OpenCL → LLVM IR)
        ↓
LLVM RISC-V Backend
        ↓
RISC-V Binary
        ↓
Vortex Runtime (vx_start, vx_ready_wait)
        ↓
Vortex Hardware/Simulator
```

---

## Path to HIP Implementation

### Option 1: Use chipStar with Vortex OpenCL (RECOMMENDED)

Since Vortex already has OpenCL support, and chipStar already has an OpenCL backend, **the path is straightforward**:

```
HIP Application
        ↓
chipStar Runtime (HIP API implementation)
        ↓
chipStar OpenCL Backend (already exists!)
        ↓
POCL on Vortex (already exists!)
        ↓
Vortex Hardware
```

**Implementation:**
1. Build chipStar with OpenCL backend
2. Point chipStar to use POCL/Vortex as OpenCL implementation
3. Done!

**Estimated effort:** 1-2 weeks (configuration + testing)

### Option 2: Direct Vortex Backend for chipStar

Create a custom chipStar backend directly for Vortex (bypass OpenCL):

```
HIP Application
        ↓
chipStar Runtime
        ↓
chipStar Vortex Backend (NEW - ~3,000 lines)
        ↓
Vortex Runtime API (vx_dev_open, vx_mem_alloc, etc.)
        ↓
Vortex Hardware
```

**Implementation:**
1. Create `CHIPBackendVortex.cc` (similar to `CHIPBackendOpenCL.cc`)
2. Implement device/memory/kernel management
3. Map HIP concepts to Vortex concepts

**Estimated effort:** 2-3 months

### Comparison

| Aspect | Option 1: Use OpenCL | Option 2: Direct Backend |
|--------|---------------------|--------------------------|
| Effort | 1-2 weeks | 2-3 months |
| Code | 0 new lines | ~3,000 lines |
| Maintenance | Low (reuse existing) | Medium (custom code) |
| Performance | Good (one extra layer) | Best (direct access) |
| Debugging | Harder (more layers) | Easier (less abstraction) |
| **Recommendation** | ✅ **Start here** | Future optimization |

---

## Hardware Capabilities

### Thread Hierarchy

Vortex matches CUDA/HIP hierarchy exactly:

| Level | Vortex | CUDA/HIP | Notes |
|-------|--------|----------|-------|
| Grid | Grid | Grid | Multiple blocks |
| Block | Group | Block | Multiple warps |
| Warp | Warp | Warp | Multiple threads (SIMT) |
| Thread | Thread | Thread | Single execution unit |

**Example Configuration:**
- 4 cores
- 4 warps per core
- 4 threads per warp
- **Total:** 64 concurrent hardware threads

### Synchronization

✅ **Hardware barriers** via `vx_barrier()`:
```c
inline void vx_barrier(int barrier_id, int num_warps) {
    __asm__ volatile (".insn r %0, 4, 0, x0, %1, %2"
                      :: "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(num_warps));
}
```

This is a **hardware instruction**, meaning:
- **Zero overhead** (like CUDA __syncthreads())
- **Not O(n) like HIP-CPU** (which uses fibers)
- **Fast and efficient**

### Memory Hierarchy

| Memory Type | Vortex | CUDA/HIP Equivalent | Speed |
|-------------|--------|---------------------|-------|
| Local Memory | `__local_mem()` | `__shared__` | Fast (on-chip) |
| Global Memory | Device buffer | Global memory | Slower (DRAM) |
| L1 Cache | Configurable | L1 cache | Very fast |
| L2 Cache | Configurable | L2 cache | Fast |
| L3 Cache | Optional | L3 cache | Medium |

**Local Memory Size:**
- Query with `VX_CAPS_LOCAL_MEM_SIZE`
- Per-group (block) allocation
- Shared by all threads in group

### ISA Extensions

```c
// Standard RISC-V
VX_ISA_STD_I    // Integer
VX_ISA_STD_M    // Multiply/Divide
VX_ISA_STD_A    // Atomics
VX_ISA_STD_F    // Single-precision floating point
VX_ISA_STD_D    // Double-precision floating point

// Vortex Extensions
VX_ISA_EXT_ICACHE    // Instruction cache
VX_ISA_EXT_DCACHE    // Data cache
VX_ISA_EXT_L2CACHE   // L2 cache
VX_ISA_EXT_L3CACHE   // L3 cache
VX_ISA_EXT_LMEM      // Local memory (shared memory)
VX_ISA_EXT_TEX       // Texture support
VX_ISA_EXT_RASTER    // Rasterization
VX_ISA_EXT_OM        // Output merger
```

---

## HIP API Mapping to Vortex

### Direct Mapping

Many HIP APIs map directly to Vortex:

| HIP API | Vortex API | Notes |
|---------|------------|-------|
| `hipGetDeviceCount()` | Check `vx_dev_open()` result | Single device |
| `hipSetDevice()` | `vx_dev_open()` | Open device |
| `hipGetDeviceProperties()` | `vx_dev_caps()` | Query capabilities |
| `hipMalloc()` | `vx_mem_alloc()` | Allocate device memory |
| `hipFree()` | `vx_mem_free()` | Free device memory |
| `hipMemcpy()` | `vx_copy_to_dev()` / `vx_copy_from_dev()` | Transfer data |
| `hipLaunchKernel()` | `vx_start()` | Launch kernel |
| `hipDeviceSynchronize()` | `vx_ready_wait()` | Wait for completion |

### Kernel-Side Mapping

| HIP | Vortex | Notes |
|-----|--------|-------|
| `threadIdx` | `threadIdx` | **Identical!** |
| `blockIdx` | `blockIdx` | **Identical!** |
| `blockDim` | `blockDim` | **Identical!** |
| `gridDim` | `gridDim` | **Identical!** |
| `__syncthreads()` | `__syncthreads()` | **Identical!** |
| `__shared__` | `__local_mem()` | Slight difference |

**The kernel API is almost identical to HIP!**

---

## Implementation Roadmap (Recommended)

### Phase 1: Use Existing OpenCL Support (Week 1-2)

1. Build chipStar with OpenCL backend
2. Configure to use POCL/Vortex
3. Test with simple HIP programs
4. Validate correctness

**Deliverable:** HIP programs running on Vortex via chipStar + OpenCL

### Phase 2: Optimize and Test (Week 3-4)

1. Run HIP test suite
2. Benchmark performance
3. Identify bottlenecks
4. Document issues

**Deliverable:** Performance baseline, test results

### Phase 3: Optional Direct Backend (Month 2-3)

If OpenCL overhead is significant:
1. Implement direct Vortex backend for chipStar
2. Bypass OpenCL layer
3. Optimize critical paths

**Deliverable:** Optimized HIP runtime for Vortex

---

## Quick Start Guide

### Prerequisites

```bash
# Vortex built and installed
source ~/vortex/build/ci/toolchain_env.sh

# chipStar built with OpenCL
cd ~/vortex_hip/chipStar
mkdir build && cd build
cmake .. -DCHIP_BUILD_OPENCL=ON
make -j$(nproc)
```

### Test HIP on Vortex

```bash
# Set chipStar to use Vortex's POCL
export OCL_ICD_VENDORS=/path/to/vortex/pocl/vendors

# Run HIP program
./hipcc vector_add.hip -o vector_add
./vector_add
```

---

## Summary

### Key Points

1. ✅ **Vortex already has most HIP requirements:**
   - Warp-based execution
   - Hardware barriers
   - Thread indexing (threadIdx, blockIdx)
   - Shared memory (__local_mem)
   - OpenCL support

2. ✅ **Easiest path: Use chipStar + OpenCL**
   - Vortex has OpenCL via POCL
   - chipStar has OpenCL backend
   - Just connect them!

3. ✅ **Kernel API is almost identical:**
   - threadIdx, blockIdx work the same
   - __syncthreads() works the same
   - Minimal porting effort

4. ✅ **Hardware support is excellent:**
   - True hardware barriers (not software like HIP-CPU)
   - Configurable warps, threads, cores
   - Local memory for shared data

### Effort Estimate

| Approach | Time | Lines of Code | Difficulty |
|----------|------|---------------|------------|
| **Use OpenCL (Recommended)** | **1-2 weeks** | **0** | **Easy** |
| Direct Vortex Backend | 2-3 months | ~3,000 | Medium |
| Full Custom Solution | 6+ months | ~15,000 | Hard |

### Next Steps

1. ✅ Build chipStar with OpenCL backend
2. ✅ Point to Vortex's POCL implementation
3. ✅ Test with simple HIP programs
4. ✅ Run full HIP test suite
5. ⚠️ Optimize if needed (direct backend)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Status:** Vortex has excellent HIP readiness
**Recommendation:** Use chipStar + OpenCL approach
