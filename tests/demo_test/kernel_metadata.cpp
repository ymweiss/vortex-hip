// Auto-generated metadata for kernel_body
#include "vortex_hip_runtime.h"
#include <stdio.h>

// Binary data symbols (created by: ld -r -b binary kernel.vxbin)
// Renamed to kernel_vxbin, kernel_vxbin_end via objcopy
extern "C" {
    extern const uint8_t kernel_vxbin[];
    extern const uint8_t kernel_vxbin_end[];
}

// Kernel handle (set by registration)
void* kernel_body_handle = nullptr;

// Metadata array
static const hipKernelArgumentMetadata kernel_body_metadata[] = {
    {.offset = 0, .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 4, .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 8, .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 12, .size = 4, .alignment = 4, .is_pointer = 0}
};

// Registration function (called at program startup)
__attribute__((constructor))
static void register_kernel_body() {
    // Calculate binary size at runtime (not compile-time)
    size_t kernel_vxbin_size = (size_t)kernel_vxbin_end - (size_t)kernel_vxbin;

    hipError_t err = __hipRegisterFunctionWithMetadata(
        &kernel_body_handle,
        "kernel_body",
        kernel_vxbin,
        kernel_vxbin_size,
        4,
        kernel_body_metadata
    );

    if (err != hipSuccess) {
        fprintf(stderr, "Failed to register kernel kernel_body: %s\n",
                hipGetErrorString(err));
    } else {
        fprintf(stderr, "Registered kernel kernel_body with %zu bytes binary and %zu arguments\n",
                kernel_vxbin_size, (size_t)4);
    }
}


