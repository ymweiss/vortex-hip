// hip_memory.cpp - HIP memory management mapped to Vortex API
// Copyright © 2024

#include "hip_vortex_runtime.h"
#include <vortex.h>
#include <cstdlib>
#include <cstring>

// Global device handle (simplified single-device model)
static vx_device_h g_vortex_device = nullptr;

hipError_t hipMalloc(void** devPtr, size_t size) {
    if (devPtr == nullptr) {
        return hipErrorInvalidValue;
    }

    if (g_vortex_device == nullptr) {
        return hipErrorNotInitialized;
    }

    vx_buffer_h buffer;
    // flags = 0 for default allocation
    int ret = vx_mem_alloc(g_vortex_device, size, 0, &buffer);
    if (ret != 0) {
        return hipErrorOutOfMemory;
    }

    // Store buffer handle as the device pointer
    *devPtr = (void*)buffer;
    return hipSuccess;
}

hipError_t hipFree(void* devPtr) {
    if (devPtr == nullptr) {
        return hipSuccess;  // Freeing NULL is valid
    }

    if (g_vortex_device == nullptr) {
        return hipErrorNotInitialized;
    }

    vx_buffer_h buffer = (vx_buffer_h)devPtr;
    int ret = vx_mem_free(buffer);
    if (ret != 0) {
        return hipErrorInvalidValue;
    }

    return hipSuccess;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return hipErrorInvalidValue;
    }

    if (g_vortex_device == nullptr) {
        return hipErrorNotInitialized;
    }

    int ret = 0;

    switch (kind) {
        case hipMemcpyHostToDevice: {
            vx_buffer_h buffer = (vx_buffer_h)dst;
            // offset = 0 (copy to start of buffer)
            ret = vx_copy_to_dev(buffer, src, 0, sizeBytes);
            break;
        }
        case hipMemcpyDeviceToHost: {
            vx_buffer_h buffer = (vx_buffer_h)src;
            // offset = 0 (copy from start of buffer)
            ret = vx_copy_from_dev(dst, buffer, 0, sizeBytes);
            break;
        }
        case hipMemcpyDeviceToDevice: {
            // TODO: Implement device-to-device copy if needed
            return hipErrorNotSupported;
        }
        case hipMemcpyHostToHost: {
            memcpy(dst, src, sizeBytes);
            return hipSuccess;
        }
        default:
            return hipErrorInvalidMemcpyDirection;
    }

    if (ret != 0) {
        return hipErrorUnknown;
    }

    return hipSuccess;
}

hipError_t hipMemset(void* devPtr, int value, size_t sizeBytes) {
    // Allocate temporary host buffer, fill it, copy to device
    // This is inefficient but works for now
    // TODO: Implement device-side memset kernel

    if (devPtr == nullptr) {
        return hipErrorInvalidValue;
    }

    if (g_vortex_device == nullptr) {
        return hipErrorNotInitialized;
    }

    void* temp = malloc(sizeBytes);
    if (temp == nullptr) {
        return hipErrorOutOfMemory;
    }

    memset(temp, value, sizeBytes);

    hipError_t err = hipMemcpy(devPtr, temp, sizeBytes, hipMemcpyHostToDevice);

    free(temp);
    return err;
}

// Internal function to get global device handle
vx_device_h __hip_get_vortex_device() {
    return g_vortex_device;
}

// Internal function to set global device handle
void __hip_set_vortex_device(vx_device_h device) {
    g_vortex_device = device;
}
