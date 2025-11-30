// hip_device.cpp - HIP device management mapped to Vortex API
// Copyright © 2024

#include "hip_vortex_runtime.h"
#include <vortex.h>
#include <cstring>

// External functions to access global device
extern vx_device_h __hip_get_vortex_device();
extern void __hip_set_vortex_device(vx_device_h device);

hipError_t hipInit(unsigned int flags) {
    (void)flags;  // Flags not used in basic implementation
    return hipSuccess;
}

hipError_t hipSetDevice(int deviceId) {
    (void)deviceId;  // Single device for now

    vx_device_h device = __hip_get_vortex_device();

    // If device not already opened, open it
    if (device == nullptr) {
        int ret = vx_dev_open(&device);
        if (ret != 0) {
            return hipErrorNoDevice;
        }
        __hip_set_vortex_device(device);
    }

    return hipSuccess;
}

hipError_t hipGetDevice(int* deviceId) {
    if (deviceId == nullptr) {
        return hipErrorInvalidValue;
    }

    // Single device model - always return 0
    *deviceId = 0;
    return hipSuccess;
}

hipError_t hipDeviceSynchronize() {
    vx_device_h device = __hip_get_vortex_device();

    if (device == nullptr) {
        return hipErrorNotInitialized;
    }

    // Wait for all operations to complete (timeout = -1 means infinite)
    int ret = vx_ready_wait(device, VX_MAX_TIMEOUT);
    if (ret != 0) {
        return hipErrorLaunchFailure;
    }

    return hipSuccess;
}

hipError_t hipDeviceReset() {
    vx_device_h device = __hip_get_vortex_device();

    if (device != nullptr) {
        vx_dev_close(device);
        __hip_set_vortex_device(nullptr);
    }

    return hipSuccess;
}

hipError_t hipGetDeviceCount(int* count) {
    if (count == nullptr) {
        return hipErrorInvalidValue;
    }

    // Single device for now
    *count = 1;
    return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    if (prop == nullptr) {
        return hipErrorInvalidValue;
    }

    if (deviceId != 0) {
        return hipErrorInvalidDevice;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        return hipErrorNotInitialized;
    }

    // Query Vortex device capabilities
    uint64_t num_cores = 0;
    uint64_t num_warps = 0;
    uint64_t num_threads = 0;
    uint64_t global_mem_size = 0;
    uint64_t local_mem_size = 0;

    vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores);
    vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps);
    vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads);
    vx_dev_caps(device, VX_CAPS_GLOBAL_MEM_SIZE, &global_mem_size);
    vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);

    // Fill in hipDeviceProp_t
    strncpy(prop->name, "Vortex RISC-V GPU", 256);
    prop->totalGlobalMem = (size_t)global_mem_size;
    prop->sharedMemPerBlock = (size_t)local_mem_size;
    prop->regsPerBlock = 0;
    prop->warpSize = (int)num_threads;
    prop->maxThreadsPerBlock = (int)(num_threads * num_warps);
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = 1;
    prop->maxThreadsDim[2] = 1;
    prop->maxGridSize[0] = 65536;
    prop->maxGridSize[1] = 65536;
    prop->maxGridSize[2] = 65536;
    prop->clockRate = 0;
    prop->multiProcessorCount = (int)num_cores;
    prop->major = 1;
    prop->minor = 0;

    return hipSuccess;
}
