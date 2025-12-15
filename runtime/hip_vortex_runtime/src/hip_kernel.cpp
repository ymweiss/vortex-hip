// hip_kernel.cpp - HIP kernel launch support mapped to Vortex API
// Copyright © 2024

#include "hip_vortex_runtime.h"
#include <vortex.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>

// External functions to access global device
extern vx_device_h __hip_get_vortex_device();
extern void __hip_set_last_error(hipError_t error);

//=============================================================================
// Kernel Argument Metadata
//=============================================================================

// Single argument metadata
struct KernelArgMeta {
    std::string name;
    std::string type;   // "ptr", "i32", "f32", etc.
    uint32_t size;      // Size in bytes
    uint32_t offset;    // Offset in args struct
    bool is_pointer;    // True if this is a device pointer
};

// Complete kernel metadata
struct KernelMetadata {
    std::string kernel_name;
    std::vector<KernelArgMeta> arguments;
    uint32_t total_args_size;
};

// Simple JSON parser for kernel metadata
// Returns true on success, false on parse error
static bool parseKernelMetadataJSON(const std::string& json, KernelMetadata& meta) {
    // Very simple JSON parsing - expects our specific format
    // Real implementation would use a proper JSON library

    meta.arguments.clear();

    // Find kernel_name
    size_t pos = json.find("\"kernel_name\"");
    if (pos == std::string::npos) return false;
    pos = json.find("\"", pos + 13);
    if (pos == std::string::npos) return false;
    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos) return false;
    meta.kernel_name = json.substr(pos + 1, end - pos - 1);

    // Find total_args_size
    pos = json.find("\"total_args_size\"");
    if (pos != std::string::npos) {
        pos = json.find(":", pos);
        if (pos != std::string::npos) {
            meta.total_args_size = std::stoul(json.substr(pos + 1));
        }
    }

    // Parse arguments array
    pos = json.find("\"arguments\"");
    if (pos == std::string::npos) return false;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return false;

    size_t array_end = json.find("]", pos);
    std::string args_section = json.substr(pos, array_end - pos);

    // Parse each argument object
    size_t arg_start = 0;
    while ((arg_start = args_section.find("{", arg_start)) != std::string::npos) {
        size_t arg_end = args_section.find("}", arg_start);
        if (arg_end == std::string::npos) break;

        std::string arg_str = args_section.substr(arg_start, arg_end - arg_start + 1);
        KernelArgMeta arg;

        // Parse name
        size_t p = arg_str.find("\"name\"");
        if (p != std::string::npos) {
            p = arg_str.find("\"", p + 6);
            size_t e = arg_str.find("\"", p + 1);
            arg.name = arg_str.substr(p + 1, e - p - 1);
        }

        // Parse type
        p = arg_str.find("\"type\"");
        if (p != std::string::npos) {
            p = arg_str.find("\"", p + 6);
            size_t e = arg_str.find("\"", p + 1);
            arg.type = arg_str.substr(p + 1, e - p - 1);
        }

        // Parse size
        p = arg_str.find("\"size\"");
        if (p != std::string::npos) {
            p = arg_str.find(":", p);
            arg.size = std::stoul(arg_str.substr(p + 1));
        }

        // Parse offset
        p = arg_str.find("\"offset\"");
        if (p != std::string::npos) {
            p = arg_str.find(":", p);
            arg.offset = std::stoul(arg_str.substr(p + 1));
        }

        // Parse is_pointer
        p = arg_str.find("\"is_pointer\"");
        if (p != std::string::npos) {
            arg.is_pointer = (arg_str.find("true", p) != std::string::npos);
        }

        meta.arguments.push_back(arg);
        arg_start = arg_end + 1;
    }

    return true;
}

// Load kernel metadata from JSON file
static bool loadKernelMetadata(const std::string& json_path, KernelMetadata& meta) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return parseKernelMetadataJSON(buffer.str(), meta);
}

//=============================================================================
// Internal Data Structures
//=============================================================================

// Information about a loaded kernel module
struct ModuleInfo {
    vx_buffer_h kernel_buffer;
    std::string filename;
};

// Information about a registered kernel (includes metadata)
struct KernelInfo {
    vx_buffer_h kernel_buffer;
    std::string name;
    std::string filename;
    KernelMetadata metadata;
    bool has_metadata;
};

// Global kernel registry - maps kernel name to info
static std::unordered_map<std::string, KernelInfo> g_kernel_registry;

// Global module registry - maps module handle to info
static std::unordered_map<hipModule_t, ModuleInfo> g_module_registry;

// Path prefix for kernel binaries (can be set via environment variable)
static const char* get_kernel_path_prefix() {
    const char* env = std::getenv("VORTEX_KERNEL_PATH");
    return env ? env : "./";
}

//=============================================================================
// Module Management
//=============================================================================

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    if (module == nullptr || fname == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Load kernel binary
    vx_buffer_h kernel_buffer;
    int ret = vx_upload_kernel_file(device, fname, &kernel_buffer);
    if (ret != 0) {
        fprintf(stderr, "hipModuleLoad: Failed to load kernel from '%s' (error %d)\n", fname, ret);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Create module info
    ModuleInfo info;
    info.kernel_buffer = kernel_buffer;
    info.filename = fname;

    // Use buffer handle as module handle
    *module = (hipModule_t)kernel_buffer;
    g_module_registry[*module] = info;

    return hipSuccess;
}

hipError_t hipModuleUnload(hipModule_t module) {
    if (module == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    auto it = g_module_registry.find(module);
    if (it == g_module_registry.end()) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // Free kernel buffer
    vx_mem_free(it->second.kernel_buffer);
    g_module_registry.erase(it);

    return hipSuccess;
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name) {
    if (function == nullptr || module == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    auto it = g_module_registry.find(module);
    if (it == g_module_registry.end()) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // For Vortex, a module contains a single kernel, so function == module
    // The name parameter is ignored but could be used for validation
    (void)name;
    *function = module;

    return hipSuccess;
}

//=============================================================================
// Kernel Registration
//=============================================================================

hipError_t hipRegisterKernel(const char* kernel_name, const char* kernel_file) {
    if (kernel_name == nullptr || kernel_file == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Load kernel binary
    vx_buffer_h kernel_buffer;
    int ret = vx_upload_kernel_file(device, kernel_file, &kernel_buffer);
    if (ret != 0) {
        fprintf(stderr, "hipRegisterKernel: Failed to load kernel '%s' from '%s' (error %d)\n",
                kernel_name, kernel_file, ret);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Register in global registry
    KernelInfo info;
    info.kernel_buffer = kernel_buffer;
    info.name = kernel_name;
    info.filename = kernel_file;
    info.has_metadata = false;

    // Try to load metadata from JSON file
    // Look for <kernel_file_base>.meta.json
    std::string meta_path = kernel_file;
    size_t dot_pos = meta_path.rfind('.');
    if (dot_pos != std::string::npos) {
        meta_path = meta_path.substr(0, dot_pos);
    }
    meta_path += ".meta.json";

    if (loadKernelMetadata(meta_path, info.metadata)) {
        info.has_metadata = true;
        printf("hipRegisterKernel: Loaded metadata for '%s' (%zu args, %u bytes)\n",
               kernel_name, info.metadata.arguments.size(), info.metadata.total_args_size);
    }

    g_kernel_registry[kernel_name] = info;

    return hipSuccess;
}

//=============================================================================
// Kernel Launch
//=============================================================================

// VortexKernelArgs and VortexKernelArgMeta are now defined in hip_vortex_runtime.h

hipError_t hipModuleLaunchKernel(
    hipFunction_t f,
    uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void** extra
) {
    (void)sharedMemBytes;  // TODO: Handle shared memory
    (void)stream;          // TODO: Handle streams
    (void)kernelParams;    // TODO: Pack kernel params
    (void)extra;           // Reserved

    if (f == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Get kernel buffer from module
    auto it = g_module_registry.find((hipModule_t)f);
    if (it == g_module_registry.end()) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }
    vx_buffer_h kernel_buffer = it->second.kernel_buffer;

    // Build argument structure
    // For hipModuleLaunchKernel, kernelParams is an array of pointers to arguments
    // We need to pack them into a contiguous buffer
    // This is a simplified implementation - real implementation would need
    // metadata about argument sizes

    VortexKernelArgs args;
    args.grid_dim[0] = gridDimX;
    args.grid_dim[1] = gridDimY;
    args.grid_dim[2] = gridDimZ;
    args.block_dim[0] = blockDimX;
    args.block_dim[1] = blockDimY;
    args.block_dim[2] = blockDimZ;

    // Upload arguments
    vx_buffer_h args_buffer;
    int ret = vx_upload_bytes(device, &args, sizeof(args), &args_buffer);
    if (ret != 0) {
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Start kernel execution
    ret = vx_start(device, kernel_buffer, args_buffer);
    if (ret != 0) {
        vx_mem_free(args_buffer);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Note: We don't wait here - that's the responsibility of hipDeviceSynchronize
    // However, we should track the args_buffer for cleanup
    // For now, we leak it - proper implementation would track pending launches

    return hipSuccess;
}

hipError_t hipLaunchKernelByName(
    const char* kernel_name,
    uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
    uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
    uint32_t sharedMemBytes,
    hipStream_t stream,
    void* args,
    size_t args_size
) {
    (void)sharedMemBytes;  // TODO: Handle shared memory
    (void)stream;          // TODO: Handle streams

    if (kernel_name == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Look up kernel in registry
    auto it = g_kernel_registry.find(kernel_name);
    vx_buffer_h kernel_buffer;

    if (it != g_kernel_registry.end()) {
        // Found in registry
        kernel_buffer = it->second.kernel_buffer;
    } else {
        // Try to load from default path: <prefix>/<kernel_name>.vxbin
        std::string kernel_path = get_kernel_path_prefix();
        kernel_path += kernel_name;
        kernel_path += ".vxbin";

        fprintf(stderr, "[HIP] Trying to load kernel: %s\n", kernel_path.c_str());
        int ret = vx_upload_kernel_file(device, kernel_path.c_str(), &kernel_buffer);
        fprintf(stderr, "[HIP] vx_upload_kernel_file returned %d\n", ret);

        // Fallback: try <prefix>/kernel.vxbin (default name from compile_hip.sh)
        if (ret != 0) {
            std::string fallback_path = get_kernel_path_prefix();
            fallback_path += "kernel.vxbin";
            fprintf(stderr, "[HIP] Fallback: trying %s\n", fallback_path.c_str());
            ret = vx_upload_kernel_file(device, fallback_path.c_str(), &kernel_buffer);
            fprintf(stderr, "[HIP] Fallback vx_upload_kernel_file returned %d\n", ret);
            if (ret == 0) {
                kernel_path = fallback_path;
            }
        }

        if (ret != 0) {
            fprintf(stderr, "hipLaunchKernelByName: Kernel '%s' not found.\n", kernel_name);
            fprintf(stderr, "  Searched: %s%s.vxbin\n", get_kernel_path_prefix(), kernel_name);
            fprintf(stderr, "  Searched: %skernel.vxbin\n", get_kernel_path_prefix());
            fprintf(stderr, "  Set VORTEX_KERNEL_PATH to specify kernel location.\n");
            __hip_set_last_error(hipErrorLaunchFailure);
            return hipErrorLaunchFailure;
        }

        // Cache for future use
        KernelInfo info;
        info.kernel_buffer = kernel_buffer;
        info.name = kernel_name;
        info.filename = kernel_path;
        g_kernel_registry[kernel_name] = info;
    }

    // Build combined argument structure with grid/block dims + user args
    size_t header_size = sizeof(VortexKernelArgs);
    size_t total_size = header_size + args_size;

    // Allocate combined buffer
    uint8_t* combined_args = new uint8_t[total_size];

    // Fill header
    VortexKernelArgs* header = (VortexKernelArgs*)combined_args;
    header->grid_dim[0] = gridDimX;
    header->grid_dim[1] = gridDimY;
    header->grid_dim[2] = gridDimZ;
    header->block_dim[0] = blockDimX;
    header->block_dim[1] = blockDimY;
    header->block_dim[2] = blockDimZ;

    // Copy user arguments
    if (args != nullptr && args_size > 0) {
        memcpy(combined_args + header_size, args, args_size);
    }

    // Upload arguments to device
    vx_buffer_h args_buffer;
    int ret = vx_upload_bytes(device, combined_args, total_size, &args_buffer);
    delete[] combined_args;

    if (ret != 0) {
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Start kernel execution
    ret = vx_start(device, kernel_buffer, args_buffer);
    if (ret != 0) {
        vx_mem_free(args_buffer);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Note: args_buffer will be leaked in this simple implementation
    // A proper implementation would track pending launches and free buffers
    // after synchronization

    return hipSuccess;
}

//=============================================================================
// vortexLaunchKernel - Launch with metadata for proper pointer conversion
//=============================================================================

// Structure for tracking device memory allocations
struct DeviceAllocation {
    vx_buffer_h buffer;
    uint64_t device_addr;
    size_t size;
};

// Global allocation tracking (simple approach - more robust would use thread-local)
static std::unordered_map<void*, DeviceAllocation> g_device_allocations;

void __hip_track_allocation(void* host_handle, vx_buffer_h buffer) {
    uint64_t dev_addr = 0;
    vx_mem_address(buffer, &dev_addr);
    g_device_allocations[host_handle] = {buffer, dev_addr, 0};
}

void __hip_untrack_allocation(void* host_handle) {
    g_device_allocations.erase(host_handle);
}

hipError_t vortexLaunchKernel(
    const char* kernel_name,
    dim3 gridDim,
    dim3 blockDim,
    const void* args,
    size_t args_size,
    const VortexKernelArgMeta* metadata,
    size_t num_args
) {
    (void)args_size; // We calculate actual size from metadata

    if (kernel_name == nullptr || args == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Load kernel (same logic as hipLaunchKernelByName)
    auto it = g_kernel_registry.find(kernel_name);
    vx_buffer_h kernel_buffer;

    if (it != g_kernel_registry.end()) {
        kernel_buffer = it->second.kernel_buffer;
    } else {
        // Try to load kernel binary
        std::string kernel_path = get_kernel_path_prefix();
        kernel_path += kernel_name;
        kernel_path += ".vxbin";

        fprintf(stderr, "[HIP] vortexLaunchKernel: Loading kernel: %s\n", kernel_path.c_str());
        int ret = vx_upload_kernel_file(device, kernel_path.c_str(), &kernel_buffer);

        // Fallback: try kernel.vxbin
        if (ret != 0) {
            std::string fallback_path = get_kernel_path_prefix();
            fallback_path += "kernel.vxbin";
            fprintf(stderr, "[HIP] Fallback: trying %s\n", fallback_path.c_str());
            ret = vx_upload_kernel_file(device, fallback_path.c_str(), &kernel_buffer);
            if (ret == 0) {
                kernel_path = fallback_path;
            }
        }

        if (ret != 0) {
            fprintf(stderr, "vortexLaunchKernel: Kernel '%s' not found.\n", kernel_name);
            __hip_set_last_error(hipErrorLaunchFailure);
            return hipErrorLaunchFailure;
        }

        // Cache for future use
        KernelInfo info;
        info.kernel_buffer = kernel_buffer;
        info.name = kernel_name;
        info.filename = kernel_path;
        g_kernel_registry[kernel_name] = info;
    }

    // Build combined argument structure:
    // Header: grid_dim[3] (12 bytes) + block_dim[3] (12 bytes) = 24 bytes
    // User args: packed with 4-byte device pointers (regardless of host pointer size)

    // Calculate device user args size
    // Device uses 4-byte pointers (RV32), scalars keep their size
    size_t device_args_size = 0;
    for (size_t i = 0; i < num_args; i++) {
        const VortexKernelArgMeta& meta = metadata[i];
        // On device, pointers are always 4 bytes
        device_args_size += meta.is_pointer ? 4 : meta.size;
    }

    size_t header_size = sizeof(VortexKernelArgs);  // 24 bytes
    size_t total_size = header_size + device_args_size;

    uint8_t* combined_args = new uint8_t[total_size];
    memset(combined_args, 0, total_size);

    // Fill header
    VortexKernelArgs* header = (VortexKernelArgs*)combined_args;
    header->grid_dim[0] = gridDim.x;
    header->grid_dim[1] = gridDim.y;
    header->grid_dim[2] = gridDim.z;
    header->block_dim[0] = blockDim.x;
    header->block_dim[1] = blockDim.y;
    header->block_dim[2] = blockDim.z;

    // Copy and convert user arguments
    // Note: meta.offset is for the HOST struct (8-byte pointers on 64-bit)
    // We compute device_offset separately for the DEVICE buffer (4-byte pointers)
    const uint8_t* src_args = (const uint8_t*)args;
    uint8_t* dst_args = combined_args + header_size;

    size_t device_offset = 0;  // Offset in device buffer
    for (size_t i = 0; i < num_args; i++) {
        const VortexKernelArgMeta& meta = metadata[i];

        if (meta.is_pointer) {
            // This is a device pointer - convert buffer handle to device address
            // Read from host struct at meta.offset (8 bytes for void*)
            void* host_handle;
            memcpy(&host_handle, src_args + meta.offset, sizeof(void*));

            uint32_t device_addr = 0;
            if (host_handle != nullptr) {
                // Look up device address from buffer handle
                vx_buffer_h buffer = (vx_buffer_h)host_handle;
                uint64_t addr64 = 0;
                vx_mem_address(buffer, &addr64);
                device_addr = (uint32_t)addr64;  // Truncate to 32-bit for RV32
            }

            // Store 4-byte device address at device_offset
            memcpy(dst_args + device_offset, &device_addr, sizeof(uint32_t));

            fprintf(stderr, "[HIP] Arg %zu: ptr handle=%p -> dev_addr=0x%08x (host_off=%u, dev_off=%zu)\n",
                    i, host_handle, device_addr, meta.offset, device_offset);

            device_offset += 4;  // Device pointers are always 4 bytes
        } else {
            // Scalar argument - copy directly
            // Read from host at meta.offset, write to device at device_offset
            memcpy(dst_args + device_offset, src_args + meta.offset, meta.size);

            fprintf(stderr, "[HIP] Arg %zu: scalar (host_off=%u, dev_off=%zu, size=%u)\n",
                    i, meta.offset, device_offset, meta.size);

            device_offset += meta.size;
        }
    }

    // Debug: print final args buffer
    fprintf(stderr, "[HIP] Args buffer (%zu bytes):\n", total_size);
    for (size_t i = 0; i < total_size && i < 48; i += 4) {
        uint32_t val;
        memcpy(&val, combined_args + i, sizeof(uint32_t));
        fprintf(stderr, "  [%2zu] 0x%08x\n", i, val);
    }

    // Upload arguments to device
    vx_buffer_h args_buffer;
    int ret = vx_upload_bytes(device, combined_args, total_size, &args_buffer);
    delete[] combined_args;

    if (ret != 0) {
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Start kernel execution
    fprintf(stderr, "[HIP] Starting kernel execution\n");
    ret = vx_start(device, kernel_buffer, args_buffer);
    if (ret != 0) {
        vx_mem_free(args_buffer);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Note: args_buffer is leaked - proper implementation would track and free after sync

    return hipSuccess;
}

//=============================================================================
// Host Library Support Functions
// Used by compiled MLIR launch wrappers for direct kernel launching
//=============================================================================

uint32_t hip_ptr_to_device_addr(const void* host_ptr) {
    if (host_ptr == nullptr) {
        return 0;
    }

    // In this runtime, hipMalloc returns vx_buffer_h as void*
    // So host_ptr IS the buffer handle
    vx_buffer_h buffer = (vx_buffer_h)host_ptr;

    uint64_t addr64 = 0;
    int ret = vx_mem_address(buffer, &addr64);
    if (ret != 0) {
        fprintf(stderr, "[HIP] hip_ptr_to_device_addr: Failed to get device address for %p\n", host_ptr);
        return 0;
    }

    // Return 32-bit address for RV32
    return (uint32_t)addr64;
}

hipError_t vortex_launch_with_args(const char* kernel_name,
                                    const void* vortex_args,
                                    size_t args_size) {
    if (kernel_name == nullptr || vortex_args == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Look up kernel in registry
    auto it = g_kernel_registry.find(kernel_name);
    vx_buffer_h kernel_buffer;

    if (it != g_kernel_registry.end()) {
        kernel_buffer = it->second.kernel_buffer;
    } else {
        // Try to load from default path
        std::string kernel_path = get_kernel_path_prefix();
        kernel_path += kernel_name;
        kernel_path += ".vxbin";

        int ret = vx_upload_kernel_file(device, kernel_path.c_str(), &kernel_buffer);

        // Fallback: try kernel.vxbin
        if (ret != 0) {
            std::string fallback_path = get_kernel_path_prefix();
            fallback_path += "kernel.vxbin";
            ret = vx_upload_kernel_file(device, fallback_path.c_str(), &kernel_buffer);
            if (ret == 0) {
                kernel_path = fallback_path;
            }
        }

        if (ret != 0) {
            fprintf(stderr, "vortex_launch_with_args: Kernel '%s' not found.\n", kernel_name);
            __hip_set_last_error(hipErrorLaunchFailure);
            return hipErrorLaunchFailure;
        }

        // Cache for future use
        KernelInfo info;
        info.kernel_buffer = kernel_buffer;
        info.name = kernel_name;
        info.filename = kernel_path;
        g_kernel_registry[kernel_name] = info;
    }

    // Upload the pre-packed argument buffer directly to device
    vx_buffer_h args_buffer;
    int ret = vx_upload_bytes(device, vortex_args, args_size, &args_buffer);
    if (ret != 0) {
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    // Launch kernel
    ret = vx_start(device, kernel_buffer, args_buffer);
    if (ret != 0) {
        vx_mem_free(args_buffer);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    return hipSuccess;
}

hipError_t vortex_launch_kernel_direct(const char* kernel_name,
                                        dim3 gridDim,
                                        dim3 blockDim,
                                        const uint32_t* device_args,
                                        size_t num_args) {
    if (kernel_name == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    vx_device_h device = __hip_get_vortex_device();
    if (device == nullptr) {
        __hip_set_last_error(hipErrorNotInitialized);
        return hipErrorNotInitialized;
    }

    // Look up kernel
    auto it = g_kernel_registry.find(kernel_name);
    vx_buffer_h kernel_buffer;

    if (it != g_kernel_registry.end()) {
        kernel_buffer = it->second.kernel_buffer;
    } else {
        // Try to load from default path
        std::string kernel_path = get_kernel_path_prefix();
        kernel_path += kernel_name;
        kernel_path += ".vxbin";

        int ret = vx_upload_kernel_file(device, kernel_path.c_str(), &kernel_buffer);
        if (ret != 0) {
            std::string fallback_path = get_kernel_path_prefix();
            fallback_path += "kernel.vxbin";
            ret = vx_upload_kernel_file(device, fallback_path.c_str(), &kernel_buffer);
            if (ret == 0) kernel_path = fallback_path;
        }

        if (ret != 0) {
            fprintf(stderr, "vortex_launch_kernel_direct: Kernel '%s' not found.\n", kernel_name);
            __hip_set_last_error(hipErrorLaunchFailure);
            return hipErrorLaunchFailure;
        }

        // Cache
        KernelInfo info;
        info.kernel_buffer = kernel_buffer;
        info.name = kernel_name;
        info.filename = kernel_path;
        g_kernel_registry[kernel_name] = info;
    }

    // Build Vortex argument buffer
    // Header (24 bytes) + user args (num_args * 4 bytes each)
    size_t total_size = sizeof(VortexKernelArgs) + num_args * sizeof(uint32_t);
    uint8_t* combined_args = new uint8_t[total_size];

    // Fill header
    VortexKernelArgs* header = (VortexKernelArgs*)combined_args;
    header->grid_dim[0] = gridDim.x;
    header->grid_dim[1] = gridDim.y;
    header->grid_dim[2] = gridDim.z;
    header->block_dim[0] = blockDim.x;
    header->block_dim[1] = blockDim.y;
    header->block_dim[2] = blockDim.z;

    // Copy user args (already 32-bit device addresses and scalars)
    if (device_args != nullptr && num_args > 0) {
        memcpy(combined_args + sizeof(VortexKernelArgs), device_args, num_args * sizeof(uint32_t));
    }

    // Upload and launch
    vx_buffer_h args_buffer;
    int ret = vx_upload_bytes(device, combined_args, total_size, &args_buffer);
    delete[] combined_args;

    if (ret != 0) {
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    ret = vx_start(device, kernel_buffer, args_buffer);
    if (ret != 0) {
        vx_mem_free(args_buffer);
        __hip_set_last_error(hipErrorLaunchFailure);
        return hipErrorLaunchFailure;
    }

    return hipSuccess;
}

hipError_t hip_get_vortex_buffer(const void* host_ptr, void** buffer) {
    if (host_ptr == nullptr || buffer == nullptr) {
        __hip_set_last_error(hipErrorInvalidValue);
        return hipErrorInvalidValue;
    }

    // In this runtime, host_ptr IS the buffer handle
    *buffer = const_cast<void*>(host_ptr);
    return hipSuccess;
}
