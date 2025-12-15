// hip_vortex_runtime.h - HIP Runtime API for Vortex
// Copyright 2024-2025 Vortex HIP Project
//
// This library provides HIP API functions that map to Vortex API calls.
// Host programs link against libhip_vortex.so to execute kernels on Vortex.
//
// Usage:
//   #include <hip/hip_runtime.h>    // or
//   #include "hip_vortex_runtime.h"
//
// Link with: -lhip_vortex -lvortex

#ifndef HIP_VORTEX_RUNTIME_H
#define HIP_VORTEX_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Types
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
// Memory Management Types
//=============================================================================

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

//=============================================================================
// Device Management
//=============================================================================

/**
 * Initialize HIP runtime
 * Maps to: vx_dev_open()
 */
hipError_t hipInit(unsigned int flags);

/**
 * Set current device
 * Maps to: vx_dev_open()
 */
hipError_t hipSetDevice(int deviceId);

/**
 * Get current device
 */
hipError_t hipGetDevice(int* deviceId);

/**
 * Wait for device to finish all work
 * Maps to: vx_ready_wait()
 */
hipError_t hipDeviceSynchronize(void);

/**
 * Reset device
 * Maps to: vx_dev_close()
 */
hipError_t hipDeviceReset(void);

/**
 * Get number of devices
 */
hipError_t hipGetDeviceCount(int* count);

/**
 * Device properties structure
 */
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

/**
 * Get device properties
 */
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);

//=============================================================================
// Memory Management
//=============================================================================

/**
 * Allocate device memory
 * Maps to: vx_mem_alloc()
 */
hipError_t hipMalloc(void** devPtr, size_t size);

/**
 * Free device memory
 * Maps to: vx_mem_free()
 */
hipError_t hipFree(void* devPtr);

/**
 * Copy memory between host and device
 * Maps to: vx_copy_to_dev() / vx_copy_from_dev()
 */
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

/**
 * Set device memory to a value
 */
hipError_t hipMemset(void* devPtr, int value, size_t sizeBytes);

//=============================================================================
// Error Handling
//=============================================================================

/**
 * Get error string for error code
 */
const char* hipGetErrorString(hipError_t error);

/**
 * Get error name for error code
 */
const char* hipGetErrorName(hipError_t error);

/**
 * Get last error and clear it
 */
hipError_t hipGetLastError(void);

/**
 * Get last error without clearing it
 */
hipError_t hipPeekAtLastError(void);

//=============================================================================
// Kernel Launch Types
//=============================================================================

/**
 * Stream handle (NULL for default stream)
 */
typedef void* hipStream_t;

/**
 * Module handle for kernel binaries
 */
typedef void* hipModule_t;

/**
 * Function handle for kernel functions
 */
typedef void* hipFunction_t;

#ifdef __cplusplus
}  // extern "C"

/**
 * dim3 structure for grid and block dimensions
 * C++ only - provides constructors for convenience
 */
struct dim3 {
    uint32_t x, y, z;

    dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1)
        : x(x_), y(y_), z(z_) {}

    dim3(const dim3& other) = default;
    dim3& operator=(const dim3& other) = default;
};

extern "C" {
#endif

//=============================================================================
// Vortex Kernel Argument Metadata
//=============================================================================

/**
 * Metadata describing a single kernel argument for Vortex marshaling.
 * Used by vortexLaunchKernel to properly convert host pointers to device addresses.
 */
typedef struct VortexKernelArgMeta {
    uint32_t offset;      // Offset in the args buffer (host layout)
    uint32_t size;        // Size in bytes (4 for i32/ptr, 8 for i64/void*)
    uint32_t is_pointer;  // 1 if this is a device pointer, 0 for scalar
} VortexKernelArgMeta;

/**
 * Vortex kernel argument header structure.
 * This is the standard format expected by Vortex kernels.
 */
typedef struct VortexKernelArgs {
    uint32_t grid_dim[3];   // Grid dimensions (12 bytes)
    uint32_t block_dim[3];  // Block dimensions (12 bytes)
    // User arguments follow at offset 24
} VortexKernelArgs;

//=============================================================================
// Kernel Launch Support
//=============================================================================

/**
 * Load a module (kernel binary) from file
 * Maps to: vx_upload_kernel_file()
 */
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);

/**
 * Unload a module
 */
hipError_t hipModuleUnload(hipModule_t module);

/**
 * Get function handle from module
 * Note: For Vortex, each module typically contains one kernel
 */
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name);

/**
 * Register a kernel binary for later launch
 */
hipError_t hipRegisterKernel(const char* kernel_name, const char* kernel_file);

/**
 * Launch kernel using module/function API
 */
hipError_t hipModuleLaunchKernel(
    hipFunction_t f,
    uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void** extra
);

/**
 * Internal kernel launch function (used by hipLaunchKernelGGL macro)
 */
hipError_t hipLaunchKernelByName(
    const char* kernel_name,
    uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes,
    hipStream_t stream,
    void* args,
    size_t args_size
);

#ifdef __cplusplus
}  // extern "C"

/**
 * Launch a Vortex kernel with explicit argument metadata.
 * This is the preferred API for launching kernels compiled by Polygeist.
 *
 * The metadata describes each kernel argument, allowing proper conversion of
 * host buffer handles (64-bit void*) to device addresses (32-bit).
 *
 * @param kernel_name   Name of the kernel (used to find .vxbin file)
 * @param gridDim       Grid dimensions
 * @param blockDim      Block dimensions
 * @param args          Pointer to packed arguments (host format)
 * @param args_size     Size of args buffer
 * @param metadata      Array describing each argument
 * @param num_args      Number of arguments in metadata array
 */
hipError_t vortexLaunchKernel(
    const char* kernel_name,
    dim3 gridDim,
    dim3 blockDim,
    const void* args,
    size_t args_size,
    const VortexKernelArgMeta* metadata,
    size_t num_args
);

//=============================================================================
// hipLaunchKernelGGL - C++ Template-based Kernel Launch (LEGACY)
//=============================================================================

/**
 * Standard HIP kernel launch macro
 *
 * DEPRECATED: This macro lacks proper argument metadata for pointer conversion.
 * Use the generated launcher functions from kernel_stubs.h instead:
 *   - launch_<kernel_name>(gridDim, blockDim, args...)
 *
 * For the new approach, include hip_vortex_host.h and the generated stubs.
 * See: Polygeist/docs/EMIT_VORTEX_WRAPPERS.md
 *
 * Usage: hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, args...)
 *
 * For Vortex, this requires that:
 * 1. The kernel has been registered with hipRegisterKernel()
 * 2. The kernel binary (.vxbin) is available
 *
 * Note: The kernel function itself is not called - only its name is used
 * to look up the registered binary.
 */
#define hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, ...) \
    do { \
        dim3 _grid = (gridDim); \
        dim3 _block = (blockDim); \
        auto _args = std::make_tuple(__VA_ARGS__); \
        hipLaunchKernelByName( \
            #kernel, \
            _grid.x, _grid.y, _grid.z, \
            _block.x, _block.y, _block.z, \
            (sharedMem), \
            (stream), \
            &_args, \
            sizeof(_args) \
        ); \
    } while(0)

/**
 * Alternative: Explicit argument struct launch
 * Use this when you have a pre-packed kernel_arg_t structure
 */
#define hipLaunchKernelWithArgs(kernel, gridDim, blockDim, sharedMem, stream, args_ptr, args_size) \
    do { \
        dim3 _grid = (gridDim); \
        dim3 _block = (blockDim); \
        hipLaunchKernelByName( \
            #kernel, \
            _grid.x, _grid.y, _grid.z, \
            _block.x, _block.y, _block.z, \
            (sharedMem), \
            (stream), \
            (args_ptr), \
            (args_size) \
        ); \
    } while(0)

// C++ template wrapper for hipMalloc to allow any pointer type
template<typename T>
inline hipError_t hipMalloc(T** ptr, size_t size) {
    return hipMalloc(reinterpret_cast<void**>(ptr), size);
}

#endif // __cplusplus

#endif // HIP_VORTEX_RUNTIME_H
