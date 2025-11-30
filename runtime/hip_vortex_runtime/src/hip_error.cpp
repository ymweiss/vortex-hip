// hip_error.cpp - HIP error handling
// Copyright © 2024

#include "hip_vortex_runtime.h"

// Global last error state
static hipError_t g_last_error = hipSuccess;

// Internal function to set last error (called by other runtime functions)
void __hip_set_last_error(hipError_t error) {
    g_last_error = error;
}

hipError_t hipGetLastError() {
    hipError_t err = g_last_error;
    g_last_error = hipSuccess;
    return err;
}

hipError_t hipPeekAtLastError() {
    return g_last_error;
}

const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess:
            return "hipSuccess";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorOutOfMemory:
            return "hipErrorOutOfMemory";
        case hipErrorNotInitialized:
            return "hipErrorNotInitialized";
        case hipErrorDeinitialized:
            return "hipErrorDeinitialized";
        case hipErrorNoDevice:
            return "hipErrorNoDevice";
        case hipErrorInvalidDevice:
            return "hipErrorInvalidDevice";
        case hipErrorInvalidMemcpyDirection:
            return "hipErrorInvalidMemcpyDirection";
        case hipErrorNotSupported:
            return "hipErrorNotSupported";
        case hipErrorLaunchFailure:
            return "hipErrorLaunchFailure";
        case hipErrorUnknown:
        default:
            return "hipErrorUnknown";
    }
}

const char* hipGetErrorName(hipError_t error) {
    // Same as hipGetErrorString for now
    return hipGetErrorString(error);
}
