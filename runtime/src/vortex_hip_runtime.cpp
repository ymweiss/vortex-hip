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

#include "vortex_hip_runtime.h"
#include <vortex.h>

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>

//=============================================================================
// Internal Data Structures
//=============================================================================

namespace {

// Global device state
struct VortexDeviceState {
    vx_device_h device = nullptr;
    int current_device_id = -1;
    bool initialized = false;
    std::mutex mutex;
};

// Memory allocation tracking
struct VortexAllocation {
    vx_buffer_h buffer;
    uint64_t device_addr;
    size_t size;
};

// Use the public metadata type internally
typedef hipKernelArgumentMetadata ArgumentMetadata;

// Kernel information
struct VortexKernelInfo {
    std::string name;
    vx_buffer_h kernel_binary;          // Uploaded kernel buffer (nullptr if not uploaded yet)
    const void* kernel_binary_data;     // Pointer to kernel binary data (for lazy upload)
    size_t binary_size;
    size_t num_args;
    std::vector<ArgumentMetadata> arg_metadata;  // Metadata for each argument
    bool uploaded;                      // True if kernel has been uploaded to device
};

// Kernel launch configuration
struct LaunchConfig {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
    std::vector<uint8_t> args;
    size_t arg_offset;
};

// Global state
static VortexDeviceState g_device_state;
static std::unordered_map<void*, VortexAllocation> g_allocations;
static std::unordered_map<const void*, VortexKernelInfo> g_kernel_registry;
static LaunchConfig g_launch_config;
static hipError_t g_last_error = hipSuccess;
static std::mutex g_mutex;

// Helper to set last error
inline void SetLastError(hipError_t error) {
    g_last_error = error;
}

// Helper to ensure device is initialized
hipError_t EnsureDeviceInitialized() {
    std::lock_guard<std::mutex> lock(g_device_state.mutex);

    if (g_device_state.initialized) {
        return hipSuccess;
    }

    int result = vx_dev_open(&g_device_state.device);
    if (result != 0) {
        SetLastError(hipErrorInitializationError);
        return hipErrorInitializationError;
    }

    g_device_state.current_device_id = 0;
    g_device_state.initialized = true;

    return hipSuccess;
}

// Helper to ensure kernel is uploaded to device (lazy upload)
hipError_t EnsureKernelUploaded(VortexKernelInfo& kernel_info) {
    if (kernel_info.uploaded) {
        return hipSuccess;
    }

    // Ensure device is initialized first
    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    // Upload kernel binary to device
    vx_buffer_h kernel_buffer;
    int result = vx_upload_kernel_bytes(g_device_state.device,
                                         kernel_info.kernel_binary_data,
                                         kernel_info.binary_size,
                                         &kernel_buffer);
    if (result != 0) {
        SetLastError(hipErrorInvalidKernelFile);
        return hipErrorInvalidKernelFile;
    }

    kernel_info.kernel_binary = kernel_buffer;
    kernel_info.uploaded = true;

    return hipSuccess;
}

}  // anonymous namespace

//=============================================================================
// Error Handling
//=============================================================================

const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";  // same as hipErrorInitializationError
        case hipErrorDeinitialized: return "hipErrorDeinitialized";
        case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration";
        case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
        case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
        case hipErrorInvalidKernelFile: return "hipErrorInvalidKernelFile";
        case hipErrorNoDevice: return "hipErrorNoDevice";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
        case hipErrorNotSupported: return "hipErrorNotSupported";
        case hipErrorUnknown: return "hipErrorUnknown";
        default: return "hipErrorUnknown";
    }
}

const char* hipGetErrorName(hipError_t error) {
    return hipGetErrorString(error);
}

hipError_t hipGetLastError(void) {
    hipError_t error = g_last_error;
    g_last_error = hipSuccess;
    return error;
}

hipError_t hipPeekAtLastError(void) {
    return g_last_error;
}

//=============================================================================
// Device Management
//=============================================================================

hipError_t hipInit(unsigned int flags) {
    if (flags != 0) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }
    return hipSuccess;
}

hipError_t hipGetDeviceCount(int* count) {
    if (!count) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // Try to open device to check availability
    vx_device_h device;
    if (vx_dev_open(&device) == 0) {
        vx_dev_close(device);
        *count = 1;
    } else {
        *count = 0;
    }

    return hipSuccess;
}

hipError_t hipSetDevice(int deviceId) {
    if (deviceId != 0) {
        SetLastError(hipErrorInvalidDevice);
        return hipErrorInvalidDevice;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    return hipSuccess;
}

hipError_t hipGetDevice(int* deviceId) {
    if (!deviceId) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    if (!g_device_state.initialized) {
        hipError_t err = EnsureDeviceInitialized();
        if (err != hipSuccess) {
            return err;
        }
    }

    *deviceId = g_device_state.current_device_id;
    return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    if (!prop) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    if (deviceId != 0) {
        SetLastError(hipErrorInvalidDevice);
        return hipErrorInvalidDevice;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    // Clear structure
    memset(prop, 0, sizeof(hipDeviceProp_t));

    // Static properties
    strncpy(prop->name, "Vortex RISC-V GPU", sizeof(prop->name) - 1);
    strncpy(prop->gcnArchName, "vortex-riscv", sizeof(prop->gcnArchName) - 1);
    prop->major = 1;
    prop->minor = 0;

    // Query hardware capabilities
    uint64_t value;

    vx_dev_caps(g_device_state.device, VX_CAPS_GLOBAL_MEM_SIZE, &value);
    prop->totalGlobalMem = value;

    vx_dev_caps(g_device_state.device, VX_CAPS_LOCAL_MEM_SIZE, &value);
    prop->sharedMemPerBlock = value;
    prop->sharedMemPerMultiprocessor = value;

    vx_dev_caps(g_device_state.device, VX_CAPS_NUM_CORES, &value);
    prop->multiProcessorCount = value;

    vx_dev_caps(g_device_state.device, VX_CAPS_NUM_THREADS, &value);
    prop->warpSize = value;

    uint64_t num_threads, num_warps;
    vx_dev_caps(g_device_state.device, VX_CAPS_NUM_THREADS, &num_threads);
    vx_dev_caps(g_device_state.device, VX_CAPS_NUM_WARPS, &num_warps);
    prop->maxThreadsPerBlock = num_threads * num_warps;
    prop->maxThreadsPerMultiProcessor = num_threads * num_warps;

    // Max dimensions (conservative defaults)
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;

    prop->maxGridSize[0] = 65535;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;

    // Feature flags
    prop->concurrentKernels = 0;
    prop->ECCEnabled = 0;
    prop->integrated = 0;
    prop->canMapHostMemory = 1;
    prop->computeMode = 0;  // Default mode
    prop->asyncEngineCount = 1;
    prop->unifiedAddressing = 0;

    vx_dev_caps(g_device_state.device, VX_CAPS_CACHE_LINE_SIZE, &value);
    prop->l2CacheSize = value * 1024;  // Convert to bytes

    // Memory bandwidth (estimate based on clock rate)
    prop->memoryClockRate = 1000000;  // 1 GHz
    prop->memoryBusWidth = 64;        // 64-bit

    // PCI info (not applicable for Vortex)
    prop->pciBusID = 0;
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;

    return hipSuccess;
}

hipError_t hipDeviceSynchronize(void) {
    if (!g_device_state.initialized) {
        hipError_t err = EnsureDeviceInitialized();
        if (err != hipSuccess) {
            return err;
        }
    }

    int result = vx_ready_wait(g_device_state.device, VX_MAX_TIMEOUT);
    if (result != 0) {
        SetLastError(hipErrorUnknown);
        return hipErrorUnknown;
    }

    return hipSuccess;
}

hipError_t hipDeviceReset(void) {
    std::lock_guard<std::mutex> lock(g_device_state.mutex);

    if (g_device_state.initialized) {
        vx_dev_close(g_device_state.device);
        g_device_state.device = nullptr;
        g_device_state.initialized = false;
        g_device_state.current_device_id = -1;
    }

    // Clear all allocations
    g_allocations.clear();
    g_kernel_registry.clear();

    return hipSuccess;
}

//=============================================================================
// Memory Management
//=============================================================================

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    if (size == 0) {
        *ptr = nullptr;
        return hipSuccess;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    vx_buffer_h buffer;
    int result = vx_mem_alloc(g_device_state.device, size, VX_MEM_READ_WRITE, &buffer);
    if (result != 0) {
        SetLastError(hipErrorOutOfMemory);
        return hipErrorOutOfMemory;
    }

    // Get device address
    uint64_t dev_addr;
    vx_mem_address(buffer, &dev_addr);

    // Store mapping (use device address as host pointer)
    void* host_ptr = reinterpret_cast<void*>(dev_addr);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_allocations[host_ptr] = {buffer, dev_addr, size};

    *ptr = host_ptr;
    return hipSuccess;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_allocations.find(ptr);
    if (it == g_allocations.end()) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    int result = vx_mem_free(it->second.buffer);
    g_allocations.erase(it);

    if (result != 0) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    return hipSuccess;
}

hipError_t hipMallocHost(void** ptr, size_t size) {
    if (!ptr) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    *ptr = malloc(size);
    if (!*ptr) {
        SetLastError(hipErrorOutOfMemory);
        return hipErrorOutOfMemory;
    }

    return hipSuccess;
}

hipError_t hipFreeHost(void* ptr) {
    if (ptr) {
        free(ptr);
    }
    return hipSuccess;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    if (!dst || !src) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    if (sizeBytes == 0) {
        return hipSuccess;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    int result = 0;
    std::lock_guard<std::mutex> lock(g_mutex);

    switch (kind) {
    case hipMemcpyHostToDevice: {
        auto it = g_allocations.find(dst);
        if (it == g_allocations.end()) {
            SetLastError(hipErrorInvalidDevicePointer);
            return hipErrorInvalidDevicePointer;
        }
        result = vx_copy_to_dev(it->second.buffer, src, 0, sizeBytes);
        break;
    }
    case hipMemcpyDeviceToHost: {
        auto it = g_allocations.find(const_cast<void*>(src));
        if (it == g_allocations.end()) {
            SetLastError(hipErrorInvalidDevicePointer);
            return hipErrorInvalidDevicePointer;
        }
        result = vx_copy_from_dev(dst, it->second.buffer, 0, sizeBytes);
        break;
    }
    case hipMemcpyDeviceToDevice: {
        // Device-to-device copy not yet implemented
        // Would need a copy kernel or DMA support
        SetLastError(hipErrorNotSupported);
        return hipErrorNotSupported;
    }
    case hipMemcpyHostToHost:
        memcpy(dst, src, sizeBytes);
        result = 0;
        break;
    default:
        SetLastError(hipErrorInvalidMemcpyDirection);
        return hipErrorInvalidMemcpyDirection;
    }

    if (result != 0) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    return hipSuccess;
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
    if (!dst) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    if (sizeBytes == 0) {
        return hipSuccess;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    // Create temporary buffer with repeated value
    std::vector<uint8_t> temp(sizeBytes, static_cast<uint8_t>(value));

    // Copy to device
    return hipMemcpy(dst, temp.data(), sizeBytes, hipMemcpyHostToDevice);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    if (!free || !total) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    uint64_t mem_free, mem_used;
    int result = vx_mem_info(g_device_state.device, &mem_free, &mem_used);
    if (result != 0) {
        SetLastError(hipErrorUnknown);
        return hipErrorUnknown;
    }

    *free = mem_free;
    *total = mem_free + mem_used;

    return hipSuccess;
}

//=============================================================================
// Kernel Execution
//=============================================================================

hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream) {
    if (!function_address) {
        SetLastError(hipErrorInvalidDeviceFunction);
        return hipErrorInvalidDeviceFunction;
    }

    hipError_t err = EnsureDeviceInitialized();
    if (err != hipSuccess) {
        return err;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    // Find kernel
    auto it = g_kernel_registry.find(function_address);
    if (it == g_kernel_registry.end()) {
        SetLastError(hipErrorInvalidDeviceFunction);
        return hipErrorInvalidDeviceFunction;
    }

    auto& kernel_info = it->second;

    // Ensure kernel is uploaded to device (lazy upload)
    err = EnsureKernelUploaded(kernel_info);
    if (err != hipSuccess) {
        return err;
    }

    // Prepare kernel arguments
    // Format: grid_dim (3x uint32), block_dim (3x uint32), shared_mem (uint64), then kernel args
    std::vector<uint8_t> arg_buffer;

    // Grid dimensions
    uint32_t grid_dims[3] = {numBlocks.x, numBlocks.y, numBlocks.z};
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(grid_dims),
                     reinterpret_cast<uint8_t*>(grid_dims) + sizeof(grid_dims));

    // Block dimensions
    uint32_t block_dims[3] = {dimBlocks.x, dimBlocks.y, dimBlocks.z};
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(block_dims),
                     reinterpret_cast<uint8_t*>(block_dims) + sizeof(block_dims));

    // Shared memory size
    uint64_t shared_mem = sharedMemBytes;
    arg_buffer.insert(arg_buffer.end(),
                     reinterpret_cast<uint8_t*>(&shared_mem),
                     reinterpret_cast<uint8_t*>(&shared_mem) + sizeof(shared_mem));

    // Marshal kernel arguments using metadata
    if (args && kernel_info.num_args > 0) {
        if (kernel_info.arg_metadata.empty()) {
            // Fallback: assume all args are 64-bit (backwards compatibility)
            for (size_t i = 0; i < kernel_info.num_args; i++) {
                uint64_t arg_value = *reinterpret_cast<uint64_t*>(args[i]);
                arg_buffer.insert(arg_buffer.end(),
                                 reinterpret_cast<uint8_t*>(&arg_value),
                                 reinterpret_cast<uint8_t*>(&arg_value) + sizeof(arg_value));
            }
        } else {
            // Use metadata for proper marshaling
            for (size_t i = 0; i < kernel_info.num_args; i++) {
                const ArgumentMetadata& meta = kernel_info.arg_metadata[i];

                // Ensure proper alignment
                size_t current_offset = arg_buffer.size();
                size_t padding = (meta.alignment - (current_offset % meta.alignment)) % meta.alignment;
                if (padding > 0) {
                    arg_buffer.resize(current_offset + padding, 0);
                }

                // Copy argument bytes with correct size
                const uint8_t* src = reinterpret_cast<const uint8_t*>(args[i]);
                arg_buffer.insert(arg_buffer.end(), src, src + meta.size);
            }
        }
    }

    // Upload arguments to device
    vx_buffer_h arg_buffer_device;
    int result = vx_upload_bytes(g_device_state.device,
                                  arg_buffer.data(),
                                  arg_buffer.size(),
                                  &arg_buffer_device);
    if (result != 0) {
        SetLastError(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Launch kernel
    result = vx_start(g_device_state.device,
                      kernel_info.kernel_binary,
                      arg_buffer_device);

    // TODO: Track arg_buffer_device for cleanup after kernel completes

    if (result != 0) {
        SetLastError(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    return hipSuccess;
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream) {
    g_launch_config.grid_dim = gridDim;
    g_launch_config.block_dim = blockDim;
    g_launch_config.shared_mem = sharedMem;
    g_launch_config.stream = stream;
    g_launch_config.args.clear();
    g_launch_config.arg_offset = 0;

    return hipSuccess;
}

hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
    if (!arg) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // Append argument to buffer
    const uint8_t* arg_bytes = reinterpret_cast<const uint8_t*>(arg);
    g_launch_config.args.insert(g_launch_config.args.end(),
                                arg_bytes,
                                arg_bytes + size);

    return hipSuccess;
}

hipError_t hipLaunchByPtr(const void* func) {
    // Convert argument buffer to array of pointers
    // This is a simplification; real implementation needs proper marshalling
    std::vector<void*> arg_ptrs;
    // TODO: Parse g_launch_config.args based on kernel metadata

    return hipLaunchKernel(func,
                           g_launch_config.grid_dim,
                           g_launch_config.block_dim,
                           arg_ptrs.data(),
                           g_launch_config.shared_mem,
                           g_launch_config.stream);
}

//=============================================================================
// Module and Function Management
//=============================================================================

hipError_t __hipRegisterFunction(void** function_address,
                                  const char* kernel_name,
                                  const void* kernel_binary,
                                  size_t kernel_size) {
    if (!function_address || !kernel_name || !kernel_binary) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // Register kernel without metadata (will use fallback marshaling)
    // NOTE: We defer kernel upload until first launch (lazy loading)
    VortexKernelInfo info;
    info.name = kernel_name;
    info.kernel_binary = nullptr;           // Not uploaded yet
    info.kernel_binary_data = kernel_binary; // Store pointer for lazy upload
    info.binary_size = kernel_size;
    info.num_args = 0;
    info.uploaded = false;                   // Mark as not uploaded
    // No metadata - will use fallback 64-bit assumption

    // Set the function handle to the kernel binary address (unique identifier)
    *function_address = const_cast<void*>(kernel_binary);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_kernel_registry[*function_address] = info;

    return hipSuccess;
}

// Extended registration with metadata
hipError_t __hipRegisterFunctionWithMetadata(void** function_address,
                                              const char* kernel_name,
                                              const void* kernel_binary,
                                              size_t kernel_size,
                                              size_t num_args,
                                              const ArgumentMetadata* arg_metadata) {
    if (!function_address || !kernel_name || !kernel_binary) {
        SetLastError(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // Register kernel with metadata
    // NOTE: We defer kernel upload until first launch (lazy loading)
    VortexKernelInfo info;
    info.name = kernel_name;
    info.kernel_binary = nullptr;           // Not uploaded yet
    info.kernel_binary_data = kernel_binary; // Store pointer for lazy upload
    info.binary_size = kernel_size;
    info.num_args = num_args;
    info.uploaded = false;                   // Mark as not uploaded

    // Copy metadata
    if (arg_metadata && num_args > 0) {
        info.arg_metadata.assign(arg_metadata, arg_metadata + num_args);
    }

    // Set the function handle to the kernel binary address (unique identifier)
    *function_address = const_cast<void*>(kernel_binary);

    std::lock_guard<std::mutex> lock(g_mutex);
    g_kernel_registry[*function_address] = info;

    return hipSuccess;
}
