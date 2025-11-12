// Manually created metadata for testing (simulates script output)
#include "vortex_hip_runtime.h"
#include <stdio.h>

// Simulated binary data
static const uint8_t vectorAdd_vxbin[] = {0x00, 0x00, 0x00, 0x00};
static const size_t vectorAdd_vxbin_size = sizeof(vectorAdd_vxbin);

static void* vectorAdd_handle = nullptr;

static const hipKernelArgumentMetadata vectorAdd_metadata[] = {
    {.offset = 0,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* a
    {.offset = 8,  .size = 8, .alignment = 8, .is_pointer = 1},  // float* b
    {.offset = 16, .size = 8, .alignment = 8, .is_pointer = 1},  // float* c
    {.offset = 24, .size = 4, .alignment = 4, .is_pointer = 0}   // int n
};

__attribute__((constructor))
static void register_vectorAdd() {
    hipError_t err = __hipRegisterFunctionWithMetadata(
        &vectorAdd_handle,
        "vectorAdd",
        vectorAdd_vxbin,
        vectorAdd_vxbin_size,
        4,
        vectorAdd_metadata
    );

    if (err != hipSuccess) {
        fprintf(stderr, "Failed to register kernel vectorAdd: %s\n",
                hipGetErrorString(err));
    } else {
        printf("✓ Kernel vectorAdd registered successfully with metadata\n");
    }
}

int main() {
    printf("=== Manual Metadata Test ===\n");
    printf("This simulates the output of hip_metadata_gen.py\n");
    printf("\n");
    printf("Expected metadata for vectorAdd(float* a, float* b, float* c, int n):\n");
    printf("  arg[0]: offset=0,  size=8, align=8, pointer=1 (float* a)\n");
    printf("  arg[1]: offset=8,  size=8, align=8, pointer=1 (float* b)\n");
    printf("  arg[2]: offset=16, size=8, align=8, pointer=1 (float* c)\n");
    printf("  arg[3]: offset=24, size=4, align=4, pointer=0 (int n)\n");
    printf("\n");

    return 0;
}
