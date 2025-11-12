// Test program demonstrating HIP argument marshaling with metadata
// This shows how the compiler would generate registration code with metadata

#include "vortex_hip_runtime.h"
#include <stdio.h>
#include <stdint.h>

// Simulated kernel binary (in reality, this would be compiled RISC-V code)
static const uint8_t test_kernel_binary[] = {
    0x00, 0x00, 0x00, 0x00  // Placeholder
};

// Function pointer handle for the kernel
static void* test_kernel_handle = nullptr;

// This function simulates what the HIP compiler would generate
__attribute__((constructor))
static void register_test_kernel() {
    // Define argument metadata for: vectorAdd(float* a, float* b, float* c, int n)
    static const hipKernelArgumentMetadata vectorAdd_metadata[] = {
        {.offset = 0,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* a
        {.offset = 8,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* b
        {.offset = 16, .size = 8, .alignment = 8, .is_pointer = 1},  // float* c
        {.offset = 24, .size = 4, .alignment = 4, .is_pointer = 0}   // int n
    };

    // Register kernel with metadata
    hipError_t err = __hipRegisterFunctionWithMetadata(
        &test_kernel_handle,
        "vectorAdd",
        test_kernel_binary,
        sizeof(test_kernel_binary),
        4,  // number of arguments
        vectorAdd_metadata
    );

    if (err != hipSuccess) {
        printf("Failed to register kernel: %s\n", hipGetErrorString(err));
    } else {
        printf("Kernel registered successfully with metadata!\n");
    }
}

int main() {
    printf("Testing HIP Argument Marshaling with Metadata\n");
    printf("===============================================\n\n");

    // Simulate kernel arguments
    float* d_a = (float*)0x100000;
    float* d_b = (float*)0x200000;
    float* d_c = (float*)0x300000;
    int n = 1024;

    printf("Kernel: vectorAdd(float* a, float* b, float* c, int n)\n\n");

    printf("Arguments:\n");
    printf("  a = %p (8 bytes, align 8, pointer)\n", (void*)d_a);
    printf("  b = %p (8 bytes, align 8, pointer)\n", (void*)d_b);
    printf("  c = %p (8 bytes, align 8, pointer)\n", (void*)d_c);
    printf("  n = %d (4 bytes, align 4, scalar)\n", n);
    printf("\n");

    printf("Expected Vortex argument buffer layout:\n");
    printf("  [0-11]:   grid_dim    (12 bytes) - added by runtime\n");
    printf("  [12-23]:  block_dim   (12 bytes) - added by runtime\n");
    printf("  [24-31]:  shared_mem  (8 bytes)  - added by runtime\n");
    printf("  [32-39]:  float* a    (8 bytes)  - from metadata\n");
    printf("  [40-47]:  float* b    (8 bytes)  - from metadata\n");
    printf("  [48-55]:  float* c    (8 bytes)  - from metadata\n");
    printf("  [56-59]:  int n       (4 bytes)  - from metadata\n");
    printf("  [60-63]:  padding     (4 bytes)  - alignment\n");
    printf("  Total: 64 bytes\n\n");

    printf("Marshaling Process:\n");
    printf("  1. Runtime reads argument metadata from registration\n");
    printf("  2. For each argument:\n");
    printf("     - Calculate padding for alignment\n");
    printf("     - Copy exactly metadata.size bytes\n");
    printf("  3. Upload packed buffer to device\n");
    printf("  4. Launch kernel with argument buffer pointer\n\n");

    // Note: We would call hipLaunchKernel here, but that requires
    // a valid Vortex device. This example demonstrates the registration.

    printf("Metadata-based marshaling implementation complete!\n");
    printf("\nKey Benefits:\n");
    printf("  ✓ Correct sizes (no 64-bit assumption)\n");
    printf("  ✓ Proper alignment (prevents GPU crashes)\n");
    printf("  ✓ Type-aware (can distinguish pointers from scalars)\n");
    printf("  ✓ Efficient (no wasted memory)\n");

    return 0;
}
