// Minimal HIP runtime header for Polygeist GPU dialect IR generation
// This header provides only kernel syntax support without runtime dependencies

#ifndef HIP_RUNTIME_POLYGEIST_H
#define HIP_RUNTIME_POLYGEIST_H

#include <stddef.h>
#include <stdint.h>

// Include Clang's CUDA builtin variables for threadIdx, blockIdx, etc.
#include <__clang_cuda_builtin_vars.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// HIP Kernel Attributes
//=============================================================================

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__   __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

//=============================================================================
// HIP Error Codes (minimal for compilation)
//=============================================================================

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
} hipError_t;

//=============================================================================
// HIP Memory Copy Types
//=============================================================================

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

//=============================================================================
// HIP Types
//=============================================================================

typedef struct dim3 {
    uint32_t x, y, z;
#ifdef __cplusplus
    __host__ __device__ dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1)
        : x(x_), y(y_), z(z_) {}
#endif
} dim3;

typedef struct hipDeviceProp_t {
    char name[256];
    int multiProcessorCount;
    // Add more fields as needed
} hipDeviceProp_t;

typedef void* hipStream_t;

//=============================================================================
// HIP Runtime API Stubs (for compilation only - not linked)
//=============================================================================

hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
hipError_t hipMemset(void* ptr, int value, size_t size);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);
const char* hipGetErrorString(hipError_t error);

//=============================================================================
// HIP Kernel Launch
//=============================================================================

// hipLaunchKernelGGL macro for kernel launches
#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...) \
    kernelName<<<numBlocks, numThreads, memPerBlock, streamId>>>(__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // HIP_RUNTIME_POLYGEIST_H
