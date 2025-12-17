// hip_runtime.h - Device-side HIP header for Vortex/Polygeist compilation
//
// This header is used during device code compilation with Polygeist.
// It provides the GPU built-in variables (threadIdx, blockIdx, etc.)
// and kernel attribute macros (__global__, __device__, etc.)
//
// Usage (in transformed source for Polygeist):
//   #ifdef __CUDA__
//   #include "hip_runtime.h"  // Device compilation
//   #endif
//
// Compile with: cgeist --cuda-lower -I runtime/device ...

#pragma once

#include <stddef.h>
#include <stdint.h>

// ------------------------------------------------------------------
// 1. Clang CUDA Built-in Variables
// Only include for device compilation (__CUDA__ defined by clang CUDA frontend)
// This provides threadIdx, blockIdx, blockDim, gridDim as GPU operations
// ------------------------------------------------------------------
#ifdef __CUDA__
#include "__clang_cuda_builtin_vars.h"
#endif

// ------------------------------------------------------------------
// 2. Kernel and Function Attributes
// Clang natively understands these in CUDA mode, but these defines
// ensure compatibility if strict checking is off.
// ------------------------------------------------------------------
#ifndef __global__
#define __global__ __attribute__((global))
#endif

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif

// ------------------------------------------------------------------
// 2a. Device-side malloc/free declarations (for cuda_wrappers/new)
// Must be declared after __device__ is defined
// ------------------------------------------------------------------
extern "C" __device__ void* malloc(size_t size);
extern "C" __device__ void free(void* ptr);

// ------------------------------------------------------------------
// 2b. Vortex warpSize Override
// Vortex has configurable threads per warp (default 4, not 32 like NVIDIA)
// Override the warpSize from __clang_cuda_builtin_vars.h if VORTEX_WARP_SIZE is defined
// ------------------------------------------------------------------
#ifdef VORTEX_WARP_SIZE
#ifdef __CUDA__
// Override the clang warpSize constant with Vortex-specific value
namespace {
  __device__ const int __vortex_warp_size = VORTEX_WARP_SIZE;
}
#define warpSize __vortex_warp_size
#endif
#endif

// ------------------------------------------------------------------
// 3. Vector Types
// HIP uses uint3/dim3 for indexing. __clang_cuda_builtin_vars.h
// defines uint3, but we define dim3 here for kernel launch syntax.
// ------------------------------------------------------------------
struct dim3 {
    unsigned int x, y, z;

    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

// ------------------------------------------------------------------
// 4. HIP Index Macros
// Map HIP-specific names to standard CUDA-style variables.
// ------------------------------------------------------------------
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x  blockIdx.x
#define hipBlockIdx_y  blockIdx.y
#define hipBlockIdx_z  blockIdx.z

#define hipBlockDim_x  blockDim.x
#define hipBlockDim_y  blockDim.y
#define hipBlockDim_z  blockDim.z

#define hipGridDim_x   gridDim.x
#define hipGridDim_y   gridDim.y
#define hipGridDim_z   gridDim.z

// ------------------------------------------------------------------
// 5. Kernel Launch Support
// Required for <<<>>> kernel launch syntax used by Polygeist
// ------------------------------------------------------------------
typedef struct cudaStream *cudaStream_t;
typedef struct hipStream *hipStream_t;

extern "C" int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                                 size_t sharedSize = 0,
                                 cudaStream_t stream = 0);

extern "C" int hipConfigureCall(dim3 gridSize, dim3 blockSize,
                                size_t sharedSize = 0,
                                hipStream_t stream = 0);

// ------------------------------------------------------------------
// 6. Error Types (minimal for device compilation)
// ------------------------------------------------------------------
typedef int hipError_t;
typedef int cudaError_t;
#define hipSuccess 0
#define cudaSuccess 0

// ------------------------------------------------------------------
// 7. printf support
// Vortex provides device printf implementation.
// printf must be callable from both host and device code.
// Using __host__ __device__ allows Clang to handle overload resolution
// correctly during CUDA compilation (both host and device jobs parse
// all code, so both contexts need a valid printf).
// ------------------------------------------------------------------
extern "C" __host__ __device__ int printf(const char*, ...);

// ------------------------------------------------------------------
// 8. Block Synchronization
// ------------------------------------------------------------------
extern "C" __device__ void __syncthreads(void);

// Memory fence intrinsics
// These ensure memory operations are visible across threads/blocks
// Implemented as inline no-ops for Vortex (can add fence.rw if needed later)
__device__ inline void __threadfence(void) {}
__device__ inline void __threadfence_block(void) {}
__device__ inline void __threadfence_system(void) {}

// ------------------------------------------------------------------
// 8a. Device Math Functions
// These provide device-side math operations for CUDA/HIP kernels.
// ------------------------------------------------------------------

// Integer min/max
__device__ inline int min(int a, int b) { return a < b ? a : b; }
__device__ inline int max(int a, int b) { return a > b ? a : b; }
__device__ inline unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }
__device__ inline unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }
__device__ inline long min(long a, long b) { return a < b ? a : b; }
__device__ inline long max(long a, long b) { return a > b ? a : b; }
__device__ inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
__device__ inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
__device__ inline long long min(long long a, long long b) { return a < b ? a : b; }
__device__ inline long long max(long long a, long long b) { return a > b ? a : b; }

// Float min/max
__device__ inline float fminf(float a, float b) { return a < b ? a : b; }
__device__ inline float fmaxf(float a, float b) { return a > b ? a : b; }

// Float math functions (declarations - implemented by libgcc/compiler-rt)
extern "C" __device__ float sqrtf(float x);
extern "C" __device__ float sinf(float x);
extern "C" __device__ float cosf(float x);
extern "C" __device__ float tanf(float x);
extern "C" __device__ float expf(float x);
extern "C" __device__ float logf(float x);
extern "C" __device__ float powf(float base, float exp);
extern "C" __device__ float fabsf(float x);
extern "C" __device__ float floorf(float x);
extern "C" __device__ float ceilf(float x);
extern "C" __device__ float roundf(float x);

// ------------------------------------------------------------------
// 9. Kernel Launch Macro (Device Compilation)
// ------------------------------------------------------------------
// When compiling with cgeist (device mode), hipLaunchKernelGGL expands
// to <<<>>> syntax. This allows CGCall.cc to capture the EXACT arguments
// being passed, making any transformations (pointer arithmetic, casts, etc.)
// visible in the generated MLIR.
//
// Usage: hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, args...)
//
// Example:
//   hipLaunchKernelGGL(mykernel, dim3(N/256), dim3(256), 0, 0, ptr + offset, n * 2);
// Expands to:
//   mykernel<<<dim3(N/256), dim3(256), 0, 0>>>(ptr + offset, n * 2);
//
// The <<<>>> syntax triggers cgeist's CUDAKernelCallExpr handling, which:
// 1. Creates gpu.launch with proper grid/block dims
// 2. Captures the actual argument expressions (not just types)
// 3. Attaches vortex.kernel_args metadata for marshalling
//
#ifdef __CUDA__
#define hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, ...) \
    kernel<<<(gridDim), (blockDim), (sharedMem), (stream)>>>(__VA_ARGS__)
#endif

// ------------------------------------------------------------------
// 10. Host Runtime API Stubs (for --emit-host-functions mode)
// ------------------------------------------------------------------
// These declarations allow HIP source files to compile without #ifdef guards
// when using cgeist --emit-host-functions. The actual implementations come
// from the host runtime library (libhip_vortex.so).
//
// Functions declared here are host-only and will be marked with the
// polygeist.host_only_func attribute by cgeist.

// Memory copy direction enum
typedef enum {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

// Memory management (internal implementation)
__host__ hipError_t __hipMalloc_impl(void** ptr, size_t size);
__host__ hipError_t hipFree(void* ptr);
__host__ hipError_t hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind);
__host__ hipError_t hipMemset(void* ptr, int value, size_t count);

// Template wrapper for hipMalloc to accept any pointer type
template <typename T>
__host__ inline hipError_t hipMalloc(T** ptr, size_t size) {
    return __hipMalloc_impl(reinterpret_cast<void**>(ptr), size);
}

// Device synchronization
__host__ hipError_t hipDeviceSynchronize(void);

// Error handling
__host__ const char* hipGetErrorString(hipError_t error);
__host__ hipError_t hipGetLastError(void);
__host__ hipError_t hipPeekAtLastError(void);

// Device management
__host__ hipError_t hipSetDevice(int deviceId);
__host__ hipError_t hipGetDevice(int* deviceId);
__host__ hipError_t hipGetDeviceCount(int* count);

// Device properties (simplified struct for compilation)
typedef struct {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int multiProcessorCount;
    int major;
    int minor;
} hipDeviceProp_t;

__host__ hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
