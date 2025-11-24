// Copyright © 2025 Vortex HIP Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIP_RUNTIME_H
#define HIP_RUNTIME_H

/**
 * HIP Runtime API Header - Vortex Backend
 *
 * This header provides inline implementations of the HIP runtime API that
 * map directly to Vortex runtime calls. This approach follows the standard
 * HIP model where backend-specific headers provide the API implementation.
 *
 * Usage:
 *   #include <hip/hip_runtime.h>
 *
 * Compilation:
 *   cgeist user_code.hip -I runtime/include --cuda-lower -S -o output.mlir
 *
 * The C preprocessor will inline these functions, so Polygeist sees direct
 * calls to vx_* functions. No HIP API awareness is needed in the compiler.
 */

#include <stddef.h>
#include <stdint.h>

// Clang CUDA built-in variables (threadIdx, blockIdx, blockDim, gridDim)
// CRITICAL: Must include this for Polygeist to recognize GPU built-ins
#include "__clang_cuda_builtin_vars.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// HIP Error Codes
//=============================================================================

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorDeinitialized = 4,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorNotSupported = 801,
    hipErrorLaunchFailure = 719,
    hipErrorUnknown = 999
} hipError_t;

//=============================================================================
// HIP Memory Transfer Kinds
//=============================================================================

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

//=============================================================================
// HIP Dimension Types (for kernel launches)
//=============================================================================

typedef struct dim3 {
    uint32_t x, y, z;
#ifdef __cplusplus
    dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z) {}
#endif
} dim3;

//=============================================================================
// Thread Index Built-ins (Device-Side)
//=============================================================================

// threadIdx, blockIdx, blockDim, gridDim are provided by __clang_cuda_builtin_vars.h
// Polygeist's --cuda-lower flag converts them to gpu.thread_id, gpu.block_id, etc.
// Our ConvertGPUToVortex pass then converts those to vx_* calls

//=============================================================================
// Device Management (Host-Side)
//=============================================================================

hipError_t hipInit(unsigned int flags);
hipError_t hipSetDevice(int deviceId);
hipError_t hipDeviceSynchronize(void);

//=============================================================================
// Memory Management (Host-Side)
//=============================================================================

hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

//=============================================================================
// Kernel Launch (Host-Side)
//=============================================================================

/**
 * Kernel Launch Syntax: kernel<<<gridDim, blockDim>>>(args...)
 *
 * This syntax is handled by Polygeist's --cuda-lower flag.
 * Polygeist converts it to gpu.launch_func in MLIR.
 * Our ConvertGPUToVortex pass then converts that to vx_upload_kernel_bytes(),
 * vx_start(), and vx_ready_wait().
 */

// Required for <<<>>> syntax support
typedef struct cudaStream *cudaStream_t;
typedef struct hipStream *hipStream_t;

#ifdef __cplusplus
extern "C" {
#endif

int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                      size_t sharedSize = 0,
                      cudaStream_t stream = 0);

int hipConfigureCall(dim3 gridSize, dim3 blockSize,
                     size_t sharedSize = 0,
                     hipStream_t stream = 0);

#ifdef __cplusplus
}
#endif

//=============================================================================
// Device Synchronization (Device-Side)
//=============================================================================

/**
 * __syncthreads() - Synchronize all threads in a block
 *
 * This is handled by Polygeist's --cuda-lower flag.
 * It gets converted to gpu.barrier in MLIR.
 * Our ConvertGPUToVortex pass then converts it to vx_barrier().
 */
#ifdef __cplusplus
extern "C" {
#endif
void __syncthreads(void);
#ifdef __cplusplus
}
#endif

//=============================================================================
// Error Handling
//=============================================================================

const char* hipGetErrorString(hipError_t error);

//=============================================================================
// Device Properties (Host-Side)
//=============================================================================

typedef struct hipDeviceProp_t {
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

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);

#ifdef __cplusplus
}
#endif

//=============================================================================
// C++ Helper Macros
//=============================================================================

#ifdef __cplusplus
// Helper for kernel attributes
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))

// hipLaunchKernelGGL - Macro for explicit kernel launch
// This expands to the <<<>>> syntax which Polygeist handles
#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...) \
    kernelName<<<numBlocks, numThreads, memPerBlock, streamId>>>(__VA_ARGS__)
#endif

#endif // HIP_RUNTIME_H
