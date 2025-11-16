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

// Include Vortex runtime API
#include <vortex.h>

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

// These are handled by Polygeist's --cuda-lower flag
// They get converted to gpu.thread_id, gpu.block_id, etc. in MLIR
// Our GPUToVortexLLVM pass then converts those to vx_* calls

#ifdef __CUDA_ARCH__  // Device code
#define threadIdx __builtin_threadIdx
#define blockIdx  __builtin_blockIdx
#define blockDim  __builtin_blockDim
#define gridDim   __builtin_gridDim
#endif

//=============================================================================
// Device Management (Host-Side)
//=============================================================================

// Global device handle for simplified API
// In a full implementation, this would use thread-local storage
extern vx_device_h __hip_vortex_device;

static inline hipError_t hipInit(unsigned int flags) {
    (void)flags;  // Unused
    return hipSuccess;
}

static inline hipError_t hipSetDevice(int deviceId) {
    if (vx_dev_open(&__hip_vortex_device) != 0) {
        return hipErrorNoDevice;
    }
    (void)deviceId;  // Single device for now
    return hipSuccess;
}

static inline hipError_t hipDeviceSynchronize(void) {
    if (vx_ready_wait(__hip_vortex_device, -1) != 0) {
        return hipErrorLaunchFailure;
    }
    return hipSuccess;
}

//=============================================================================
// Memory Management (Host-Side)
//=============================================================================

/**
 * hipMalloc - Allocate device memory
 *
 * Maps to: vx_mem_alloc()
 *
 * Example transformation:
 *   Source:  hipMalloc(&d_ptr, 1024);
 *   Inline:  vx_mem_alloc(__hip_vortex_device, 1024, (uint64_t*)&d_ptr);
 *   Polygeist sees: func.call @vx_mem_alloc(...)
 */
static inline hipError_t hipMalloc(void** ptr, size_t size) {
    if (ptr == NULL) {
        return hipErrorInvalidValue;
    }

    uint64_t dev_ptr = 0;
    if (vx_mem_alloc(__hip_vortex_device, size, &dev_ptr) != 0) {
        return hipErrorOutOfMemory;
    }

    *ptr = (void*)(uintptr_t)dev_ptr;
    return hipSuccess;
}

/**
 * hipFree - Free device memory
 *
 * Maps to: vx_mem_free()
 */
static inline hipError_t hipFree(void* ptr) {
    if (ptr == NULL) {
        return hipSuccess;  // Freeing NULL is valid
    }

    uint64_t dev_ptr = (uint64_t)(uintptr_t)ptr;
    if (vx_mem_free(__hip_vortex_device, dev_ptr) != 0) {
        return hipErrorInvalidValue;
    }

    return hipSuccess;
}

/**
 * hipMemcpy - Copy memory between host and device
 *
 * Maps to:
 *   - vx_copy_to_dev() for Host → Device
 *   - vx_copy_from_dev() for Device → Host
 *
 * Example transformation:
 *   Source:  hipMemcpy(d_ptr, h_ptr, 1024, hipMemcpyHostToDevice);
 *   Inline:  vx_copy_to_dev(__hip_vortex_device, (uint64_t)d_ptr, h_ptr, 1024);
 *   Polygeist sees: func.call @vx_copy_to_dev(...)
 */
static inline hipError_t hipMemcpy(void* dst, const void* src,
                                    size_t sizeBytes, hipMemcpyKind kind) {
    if (dst == NULL || src == NULL) {
        return hipErrorInvalidValue;
    }

    int result = 0;

    switch (kind) {
        case hipMemcpyHostToDevice: {
            uint64_t dev_dst = (uint64_t)(uintptr_t)dst;
            result = vx_copy_to_dev(__hip_vortex_device, dev_dst, src, sizeBytes);
            break;
        }
        case hipMemcpyDeviceToHost: {
            uint64_t dev_src = (uint64_t)(uintptr_t)src;
            result = vx_copy_from_dev(__hip_vortex_device, dst, dev_src, sizeBytes);
            break;
        }
        case hipMemcpyDeviceToDevice:
            // TODO: Implement device-to-device copy if Vortex supports it
            return hipErrorNotSupported;
        case hipMemcpyHostToHost:
            // Use standard memcpy for host-to-host
            memcpy(dst, src, sizeBytes);
            return hipSuccess;
        default:
            return hipErrorInvalidMemcpyDirection;
    }

    if (result != 0) {
        return hipErrorUnknown;
    }

    return hipSuccess;
}

//=============================================================================
// Kernel Launch (Host-Side)
//=============================================================================

/**
 * Kernel Launch Syntax: kernel<<<gridDim, blockDim>>>(args...)
 *
 * This syntax is handled by Polygeist's --cuda-lower flag.
 * Polygeist converts it to gpu.launch_func in MLIR.
 * Our GPUToVortexLLVM pass then converts that to vx_upload_kernel_bytes(),
 * vx_start(), and vx_ready_wait().
 *
 * No inline implementation needed here - handled by compiler transformation.
 */

//=============================================================================
// Device Synchronization (Device-Side)
//=============================================================================

/**
 * __syncthreads() - Synchronize all threads in a block
 *
 * This is handled by Polygeist's --cuda-lower flag.
 * It gets converted to gpu.barrier in MLIR.
 * Our GPUToVortexLLVM pass then converts it to vx_barrier().
 *
 * No inline implementation needed - handled by compiler.
 */
#ifdef __CUDA_ARCH__
void __syncthreads(void);
#endif

//=============================================================================
// Error Handling
//=============================================================================

static inline const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";
        case hipErrorDeinitialized: return "hipErrorDeinitialized";
        case hipErrorNoDevice: return "hipErrorNoDevice";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        default: return "hipErrorUnknown";
    }
}

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
#endif

#endif // HIP_RUNTIME_H
