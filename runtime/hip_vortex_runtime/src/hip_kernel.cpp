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

/**
 * Vortex kernel argument structure
 * This is the standard format expected by Vortex kernels
 */
struct VortexKernelArgs {
    // Grid and block dimensions
    uint32_t grid_dim[3];
    uint32_t block_dim[3];

    // User arguments follow (variable size)
    // The actual arguments are appended after this header
};

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
        // Try to load from default path
        std::string kernel_path = get_kernel_path_prefix();
        kernel_path += kernel_name;
        kernel_path += ".vxbin";

        int ret = vx_upload_kernel_file(device, kernel_path.c_str(), &kernel_buffer);
        if (ret != 0) {
            fprintf(stderr, "hipLaunchKernelByName: Kernel '%s' not registered and not found at '%s'\n",
                    kernel_name, kernel_path.c_str());
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
