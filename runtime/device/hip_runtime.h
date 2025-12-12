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
// 7. Device-side printf support
// ------------------------------------------------------------------
extern "C" __device__ int printf(const char*, ...);

// ------------------------------------------------------------------
// 8. Block Synchronization
// ------------------------------------------------------------------
extern "C" __device__ void __syncthreads(void);
