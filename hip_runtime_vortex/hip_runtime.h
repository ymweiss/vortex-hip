#pragma once

#include <stddef.h>

// ------------------------------------------------------------------
// 1. Clang CUDA Built-in Variables
// CRITICAL: Must use Clang's builtin header for Polygeist to recognize
// threadIdx, blockIdx, etc. as GPU operations (not regular variables)
// ------------------------------------------------------------------
#include "__clang_cuda_builtin_vars.h"

// ------------------------------------------------------------------
// 2. Attributes
// Clang natively understands __global__, etc. when in HIP mode,
// but these defines ensure compatibility if strict checking is off.
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

// ------------------------------------------------------------------
// 3. Vector Types
// HIP uses uint3/dim3 for indexing. We need basic structs.
// Note: __clang_cuda_builtin_vars.h defines uint3 already,
// but we define dim3 here for kernel launch syntax.
// ------------------------------------------------------------------
struct dim3 {
    unsigned int x, y, z;

    // dim3 often has a constructor in real headers
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

// ------------------------------------------------------------------
// 4. HIP Macros
// Map the HIP-specific names to the standard CUDA-style variables.
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
// These declarations are required for the <<<>>> kernel launch syntax
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
// 6. Runtime Function Stubs
// For actual compilation (not Polygeist), these provide minimal stubs
// ------------------------------------------------------------------
typedef int hipError_t;
typedef int cudaError_t;
#define hipSuccess 0
#define cudaSuccess 0