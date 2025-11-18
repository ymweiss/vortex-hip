// HIP Runtime API - Vortex Backend
// Maps HIP API calls to Vortex runtime calls
// This header is included instead of the standard HIP runtime

#ifndef HIP_RUNTIME_VORTEX_H
#define HIP_RUNTIME_VORTEX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
// Forward declarations of Vortex runtime functions
///////////////////////////////////////////////////////////////////////////////

// Device management
typedef void* vx_device_h;
int vx_dev_open(vx_device_h* hdevice);
int vx_dev_close(vx_device_h hdevice);

// Memory management
typedef void* vx_buffer_h;
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);
int vx_mem_free(vx_buffer_h hbuffer);
int vx_mem_address(vx_buffer_h hbuffer, uint64_t* dev_addr);

// Memory transfer
int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size);
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size);

// Kernel execution
int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size, vx_buffer_h* hbuffer);
int vx_start(vx_device_h hdevice, vx_buffer_h hkernel, vx_buffer_h hargs);
int vx_ready_wait(vx_device_h hdevice, uint64_t timeout);

///////////////////////////////////////////////////////////////////////////////
// HIP API Types
///////////////////////////////////////////////////////////////////////////////

typedef int hipError_t;
typedef void* hipStream_t;

// Error codes
#define hipSuccess 0
#define hipErrorMemoryAllocation 1
#define hipErrorInvalidValue 2

// Memory copy kinds
typedef enum {
    hipMemcpyHostToDevice,
    hipMemcpyDeviceToHost,
    hipMemcpyDeviceToDevice,
    hipMemcpyHostToHost
} hipMemcpyKind;

// Dimension types
struct dim3 {
    uint32_t x, y, z;
#ifdef __cplusplus
    dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z) {}
#endif
};

typedef struct dim3 dim3;

// Device properties
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
} hipDeviceProp_t;

///////////////////////////////////////////////////////////////////////////////
// Global device handle (simplified - single device)
///////////////////////////////////////////////////////////////////////////////

static vx_device_h g_vortex_device = NULL;

static inline vx_device_h vx_get_device() {
    if (g_vortex_device == NULL) {
        vx_dev_open(&g_vortex_device);
    }
    return g_vortex_device;
}

///////////////////////////////////////////////////////////////////////////////
// HIP API Implementation - Memory Management
///////////////////////////////////////////////////////////////////////////////

static inline hipError_t hipMalloc(void** ptr, size_t size) {
    vx_buffer_h buffer;
    int ret = vx_mem_alloc(vx_get_device(), size, 0, &buffer);
    if (ret != 0) return hipErrorMemoryAllocation;

    uint64_t dev_addr;
    ret = vx_mem_address(buffer, &dev_addr);
    if (ret != 0) return hipErrorMemoryAllocation;

    *ptr = (void*)dev_addr;
    return hipSuccess;
}

static inline hipError_t hipFree(void* ptr) {
    // Note: In real implementation, need to track buffer handles
    // For now, simplified
    return hipSuccess;
}

static inline hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    // Note: Simplified implementation
    // In real version, need to track buffer handles and call appropriate vx_copy_* function
    return hipSuccess;
}

static inline hipError_t hipMemset(void* dst, int value, size_t size) {
    // TODO: Implement using Vortex memory operations
    return hipSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// HIP API Implementation - Device Management
///////////////////////////////////////////////////////////////////////////////

static inline hipError_t hipSetDevice(int deviceId) {
    // Simplified: single device
    return hipSuccess;
}

static inline hipError_t hipGetDevice(int* deviceId) {
    *deviceId = 0;
    return hipSuccess;
}

static inline hipError_t hipGetDeviceCount(int* count) {
    *count = 1;
    return hipSuccess;
}

static inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    // Fill with Vortex device properties
    prop->totalGlobalMem = 1024 * 1024 * 1024; // 1GB
    prop->sharedMemPerBlock = 16384; // 16KB
    prop->warpSize = 32;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->multiProcessorCount = 4;
    return hipSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// HIP API Implementation - Synchronization
///////////////////////////////////////////////////////////////////////////////

static inline hipError_t hipDeviceSynchronize() {
    return vx_ready_wait(vx_get_device(), -1);
}

static inline hipError_t hipStreamSynchronize(hipStream_t stream) {
    return hipDeviceSynchronize();
}

///////////////////////////////////////////////////////////////////////////////
// HIP API Implementation - Error Handling
///////////////////////////////////////////////////////////////////////////////

static inline const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorMemoryAllocation: return "hipErrorMemoryAllocation";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        default: return "hipErrorUnknown";
    }
}

static inline hipError_t hipGetLastError() {
    return hipSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel Launch (handled by Polygeist --cuda-lower)
///////////////////////////////////////////////////////////////////////////////

// The <<<>>> syntax is handled by Polygeist's --cuda-lower flag
// Polygeist will generate gpu.launch_func operations
// Our custom pass will convert those to vx_upload_kernel_bytes() + vx_start()

///////////////////////////////////////////////////////////////////////////////
// Device-side built-ins (recognized by Polygeist)
///////////////////////////////////////////////////////////////////////////////

// These are recognized by Polygeist and converted to GPU dialect operations:
// - threadIdx, blockIdx, blockDim, gridDim
// - __syncthreads()
// - __shared__ memory

#ifdef __cplusplus
}
#endif

#endif // HIP_RUNTIME_VORTEX_H
