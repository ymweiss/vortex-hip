// Manually created metadata for vecadd kernel
// This demonstrates what the compiler WILL generate in Phase 2
// For Phase 1, we create this manually since DWARF parsing is limited

#include "vortex_hip_runtime.h"
#include <stdio.h>

// Binary data symbols (created by: ld -r -b binary kernel.vxbin)
extern "C" {
    extern const uint8_t kernel_vxbin[];
    extern const uint8_t kernel_vxbin_end[];
}

// Calculate binary size
static const size_t kernel_vxbin_size =
    (size_t)(&kernel_vxbin_end[0]) - (size_t)(&kernel_vxbin[0]);

// Kernel handle (set by registration, used by main.cpp)
void* vecadd_handle = nullptr;

// Metadata array for: vecadd(float* a, float* b, float* c, uint32_t n)
// Architecture: RV32 (32-bit pointers)
static const hipKernelArgumentMetadata vecadd_metadata[] = {
    {.offset = 0,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* a
    {.offset = 4,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* b
    {.offset = 8,  .size = 4, .alignment = 4, .is_pointer = 1},  // float* c
    {.offset = 12, .size = 4, .alignment = 4, .is_pointer = 0}   // uint32_t n
};

// Registration function (called at program startup)
__attribute__((constructor))
static void register_vecadd() {
    hipError_t err = __hipRegisterFunctionWithMetadata(
        &vecadd_handle,
        "vecadd",
        kernel_vxbin,
        kernel_vxbin_size,
        4,
        vecadd_metadata
    );

    if (err != hipSuccess) {
        fprintf(stderr, "Failed to register kernel vecadd: %s\n",
                hipGetErrorString(err));
    } else {
        fprintf(stderr, "Registered kernel vecadd with %zu bytes binary and 4 arguments\n",
                kernel_vxbin_size);
    }
}
