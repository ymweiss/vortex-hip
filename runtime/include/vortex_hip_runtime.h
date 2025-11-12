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

#ifndef VORTEX_HIP_RUNTIME_H
#define VORTEX_HIP_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

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
    hipErrorInitializationError = 3,  // Alias for hipErrorNotInitialized
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,
    hipErrorInvalidMemcpyDirection = 21,
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,
    hipErrorInvalidDevice = 101,
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled = 704,
    hipErrorPeerAccessNotEnabled = 705,
    hipErrorSetOnActiveProcess = 708,
    hipErrorContextIsDestroyed = 709,
    hipErrorAssert = 710,
    hipErrorHostMemoryAlreadyRegistered = 712,
    hipErrorHostMemoryNotRegistered = 713,
    hipErrorLaunchFailure = 719,
    hipErrorCooperativeLaunchTooLarge = 720,
    hipErrorNotSupported = 801,
    hipErrorStreamCaptureUnsupported = 900,
    hipErrorStreamCaptureInvalidated = 901,
    hipErrorStreamCaptureMerge = 902,
    hipErrorStreamCaptureUnmatched = 903,
    hipErrorStreamCaptureUnjoined = 904,
    hipErrorStreamCaptureIsolation = 905,
    hipErrorStreamCaptureImplicit = 906,
    hipErrorCapturedEvent = 907,
    hipErrorStreamCaptureWrongThread = 908,
    hipErrorGraphExecUpdateFailure = 910,
    hipErrorUnknown = 999
} hipError_t;

//=============================================================================
// HIP Memory Copy Kinds
//=============================================================================

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

//=============================================================================
// HIP Dimension Types
//=============================================================================

typedef struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
#ifdef __cplusplus
    dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z) {}
#endif
} dim3;

//=============================================================================
// HIP Stream Type
//=============================================================================

typedef void* hipStream_t;
#define hipStreamDefault ((hipStream_t)0)

//=============================================================================
// HIP Device Properties
//=============================================================================

typedef struct hipDeviceProp_t {
    char name[256];                     // Device name
    size_t totalGlobalMem;              // Global memory available on device in bytes
    size_t sharedMemPerBlock;           // Shared memory available per block in bytes
    int regsPerBlock;                   // Registers available per block
    int warpSize;                       // Warp size in threads
    size_t memPitch;                    // Maximum pitch in bytes allowed by memory copies
    int maxThreadsPerBlock;             // Maximum number of threads per block
    int maxThreadsDim[3];               // Maximum size of each dimension of a block
    int maxGridSize[3];                 // Maximum size of each dimension of a grid
    int clockRate;                      // Clock frequency in kilohertz
    size_t totalConstMem;               // Constant memory available on device in bytes
    int major;                          // Major compute capability
    int minor;                          // Minor compute capability
    size_t textureAlignment;            // Alignment requirement for textures
    int deviceOverlap;                  // Device can concurrently copy memory and execute a kernel
    int multiProcessorCount;            // Number of multiprocessors on device
    int kernelExecTimeoutEnabled;       // Run time limit on kernels
    int integrated;                     // Device is integrated as opposed to discrete
    int canMapHostMemory;               // Device can map host memory with hipHostMalloc/hipHostGetDevicePointer
    int computeMode;                    // Compute mode
    int maxTexture1D;                   // Maximum 1D texture size
    int maxTexture2D[2];                // Maximum 2D texture dimensions
    int maxTexture3D[3];                // Maximum 3D texture dimensions
    int concurrentKernels;              // Device can possibly execute multiple kernels concurrently
    int ECCEnabled;                     // Device has ECC support enabled
    int pciBusID;                       // PCI bus ID of the device
    int pciDeviceID;                    // PCI device ID of the device
    int pciDomainID;                    // PCI domain ID of the device
    int tccDriver;                      // 1 if device is a Tesla device using TCC driver, 0 otherwise
    int asyncEngineCount;               // Number of asynchronous engines
    int unifiedAddressing;              // Device shares a unified address space with the host
    int memoryClockRate;                // Peak memory clock frequency in kilohertz
    int memoryBusWidth;                 // Global memory bus width in bits
    int l2CacheSize;                    // Size of L2 cache in bytes
    int maxThreadsPerMultiProcessor;    // Maximum resident threads per multiprocessor
    int streamPrioritiesSupported;      // Device supports stream priorities
    int globalL1CacheSupported;         // Device supports caching globals in L1
    int localL1CacheSupported;          // Device supports caching locals in L1
    size_t sharedMemPerMultiprocessor;  // Shared memory available per multiprocessor in bytes
    int regsPerMultiprocessor;          // Registers available per multiprocessor
    int managedMemory;                  // Device supports allocating managed memory on this system
    int isMultiGpuBoard;                // Device is on a multi-GPU board
    int multiGpuBoardGroupID;           // Unique identifier for a group of devices on the same multi-GPU board
    int singleToDoublePrecisionPerfRatio; // Ratio of single precision performance to double precision
    int pageableMemoryAccess;           // Device supports coherently accessing pageable memory
    int concurrentManagedAccess;        // Device can coherently access managed memory concurrently with the CPU
    char gcnArchName[256];              // AMD GCN architecture name
    int cooperativeLaunch;              // Device supports cooperative kernel launch
    int cooperativeMultiDeviceLaunch;   // Device supports cooperative multi-device kernel launch
} hipDeviceProp_t;

//=============================================================================
// HIP UUID
//=============================================================================

typedef struct hipUUID_t {
    char bytes[16];
} hipUUID_t;

//=============================================================================
// Device Management
//=============================================================================

/**
 * @brief Initialize the HIP runtime
 * @param flags Must be 0
 * @return hipSuccess on success
 */
hipError_t hipInit(unsigned int flags);

/**
 * @brief Get number of available devices
 * @param count Pointer to store device count
 * @return hipSuccess on success
 */
hipError_t hipGetDeviceCount(int* count);

/**
 * @brief Set the active device
 * @param deviceId Device ID (0-based)
 * @return hipSuccess on success
 */
hipError_t hipSetDevice(int deviceId);

/**
 * @brief Get the currently active device
 * @param deviceId Pointer to store device ID
 * @return hipSuccess on success
 */
hipError_t hipGetDevice(int* deviceId);

/**
 * @brief Get device properties
 * @param prop Pointer to device properties structure
 * @param deviceId Device ID
 * @return hipSuccess on success
 */
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);

/**
 * @brief Wait for device to finish all operations
 * @return hipSuccess on success
 */
hipError_t hipDeviceSynchronize(void);

/**
 * @brief Reset the device
 * @return hipSuccess on success
 */
hipError_t hipDeviceReset(void);

//=============================================================================
// Memory Management
//=============================================================================

/**
 * @brief Allocate memory on the device
 * @param ptr Pointer to store allocated device pointer
 * @param size Size in bytes to allocate
 * @return hipSuccess on success
 */
hipError_t hipMalloc(void** ptr, size_t size);

/**
 * @brief Free device memory
 * @param ptr Device pointer to free
 * @return hipSuccess on success
 */
hipError_t hipFree(void* ptr);

/**
 * @brief Allocate pinned host memory
 * @param ptr Pointer to store allocated host pointer
 * @param size Size in bytes to allocate
 * @return hipSuccess on success
 */
hipError_t hipMallocHost(void** ptr, size_t size);

/**
 * @brief Free pinned host memory
 * @param ptr Host pointer to free
 * @return hipSuccess on success
 */
hipError_t hipFreeHost(void* ptr);

/**
 * @brief Copy memory between host and device
 * @param dst Destination pointer
 * @param src Source pointer
 * @param sizeBytes Size in bytes to copy
 * @param kind Direction of copy (hipMemcpyHostToDevice, etc.)
 * @return hipSuccess on success
 */
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

/**
 * @brief Set device memory to a value
 * @param dst Device pointer
 * @param value Value to set (0-255)
 * @param sizeBytes Size in bytes
 * @return hipSuccess on success
 */
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);

/**
 * @brief Get available and total device memory
 * @param free Pointer to store free memory
 * @param total Pointer to store total memory
 * @return hipSuccess on success
 */
hipError_t hipMemGetInfo(size_t* free, size_t* total);

//=============================================================================
// Kernel Execution
//=============================================================================

/**
 * @brief Launch a kernel on the device
 * @param function_address Pointer to kernel function
 * @param numBlocks Grid dimensions
 * @param dimBlocks Block dimensions
 * @param args Array of pointers to kernel arguments
 * @param sharedMemBytes Shared memory size in bytes
 * @param stream Stream to execute on (0 for default)
 * @return hipSuccess on success
 */
hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream);

/**
 * @brief Configure next kernel launch (used by <<<>>> syntax)
 * @param gridDim Grid dimensions
 * @param blockDim Block dimensions
 * @param sharedMem Shared memory size
 * @param stream Stream
 * @return hipSuccess on success
 */
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);

/**
 * @brief Set up kernel argument (used by <<<>>> syntax)
 * @param arg Pointer to argument
 * @param size Size of argument
 * @param offset Offset in argument list
 * @return hipSuccess on success
 */
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);

/**
 * @brief Launch kernel configured by hipConfigureCall and hipSetupArgument
 * @param func Kernel function pointer
 * @return hipSuccess on success
 */
hipError_t hipLaunchByPtr(const void* func);

//=============================================================================
// Error Handling
//=============================================================================

/**
 * @brief Get the last error and clear it
 * @return Last error code
 */
hipError_t hipGetLastError(void);

/**
 * @brief Peek at the last error without clearing it
 * @return Last error code
 */
hipError_t hipPeekAtLastError(void);

/**
 * @brief Get error string from error code
 * @param error Error code
 * @return String describing the error
 */
const char* hipGetErrorString(hipError_t error);

/**
 * @brief Get error name from error code
 * @param error Error code
 * @return String name of the error
 */
const char* hipGetErrorName(hipError_t error);

//=============================================================================
// Module and Function Management
//=============================================================================

/**
 * @brief Argument metadata structure for kernel registration
 */
typedef struct {
    size_t offset;        // Offset in argument buffer
    size_t size;          // Size in bytes
    size_t alignment;     // Required alignment
    int is_pointer;       // Non-zero if this is a pointer type
} hipKernelArgumentMetadata;

/**
 * @brief Register a kernel function (called by compiler-generated code)
 * @param function_address Host-side function pointer
 * @param kernel_name Kernel name
 * @param kernel_binary Device kernel binary
 * @param kernel_size Size of kernel binary
 * @return hipSuccess on success
 */
hipError_t __hipRegisterFunction(void** function_address,
                                  const char* kernel_name,
                                  const void* kernel_binary,
                                  size_t kernel_size);

/**
 * @brief Register a kernel function with argument metadata
 * @param function_address Host-side function pointer
 * @param kernel_name Kernel name
 * @param kernel_binary Device kernel binary
 * @param kernel_size Size of kernel binary
 * @param num_args Number of kernel arguments
 * @param arg_metadata Array of argument metadata
 * @return hipSuccess on success
 */
hipError_t __hipRegisterFunctionWithMetadata(void** function_address,
                                              const char* kernel_name,
                                              const void* kernel_binary,
                                              size_t kernel_size,
                                              size_t num_args,
                                              const hipKernelArgumentMetadata* arg_metadata);

//=============================================================================
// C++ Helper Macros
//=============================================================================

#ifdef __cplusplus
}  // extern "C"

// Kernel launch syntax: kernel<<<grid, block, sharedMem, stream>>>(args...)
#define hipLaunchKernelGGL(F, G, B, S, ST, ...) \
    do { \
        hipConfigureCall(G, B, S, ST); \
        F(__VA_ARGS__); \
    } while(0)

#endif  // __cplusplus

#endif  // VORTEX_HIP_RUNTIME_H
