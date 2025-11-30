// hip_vortex_runtime.h - Minimal HIP runtime for Vortex
// Copyright © 2024
//
// This library provides HIP API functions that map to Vortex API calls.
// Host programs link against this library to execute kernels on Vortex.

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
// Memory Management
//=============================================================================

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

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
// Device Management
//=============================================================================

/**
 * Initialize HIP runtime
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
hipError_t hipDeviceSynchronize();

/**
 * Reset device
 * Maps to: vx_dev_close()
 */
hipError_t hipDeviceReset();

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
 * This associates a kernel name with a loaded binary
 */
hipError_t hipRegisterKernel(const char* kernel_name, const char* kernel_file);

/**
 * Launch kernel using module/function API
 *
 * @param f         Function handle from hipModuleGetFunction
 * @param gridDimX  Grid dimension X
 * @param gridDimY  Grid dimension Y
 * @param gridDimZ  Grid dimension Z
 * @param blockDimX Block dimension X
 * @param blockDimY Block dimension Y
 * @param blockDimZ Block dimension Z
 * @param sharedMemBytes Shared memory size
 * @param stream    Stream (NULL for default)
 * @param kernelParams Array of pointers to kernel arguments
 * @param extra     Reserved, must be NULL
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
 *
 * @param kernel_name   Name of the kernel to launch
 * @param gridDimX      Grid dimension X
 * @param gridDimY      Grid dimension Y
 * @param gridDimZ      Grid dimension Z
 * @param blockDimX     Block dimension X
 * @param blockDimY     Block dimension Y
 * @param blockDimZ     Block dimension Z
 * @param sharedMemBytes Shared memory size
 * @param stream        Stream (NULL for default)
 * @param args          Pointer to argument data blob
 * @param args_size     Size of argument data
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

//=============================================================================
// hipLaunchKernelGGL - C++ Template-based Kernel Launch
//=============================================================================

/**
 * Standard HIP kernel launch macro
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

#endif // __cplusplus

#endif // HIP_VORTEX_RUNTIME_H
