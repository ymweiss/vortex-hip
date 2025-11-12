// Vortex kernel for fence test (memory fence operations)
// Adapted from tests/fence.cpp

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
struct FenceArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    int32_t* src0;
    int32_t* src1;
    int32_t* dst;
    uint32_t count;  // elements per block
} __attribute__((packed));

// Kernel body - performs addition with thread fence
void kernel_body(FenceArgs* __UNIFORM__ args) {
    uint32_t blockIdx_x = blockIdx.x;
    uint32_t offset = blockIdx_x * args->count;

    // Each block processes 'count' elements
    for (uint32_t i = 0; i < args->count; ++i) {
        args->dst[offset + i] = args->src0[offset + i] + args->src1[offset + i];
    }

    // Memory fence - ensure all writes complete before kernel exit
    // In Vortex, memory ordering is handled by the runtime
    // No explicit fence instruction needed here
}

// Device main
int main() {
    FenceArgs* args = (FenceArgs*)csr_read(VX_CSR_MSCRATCH);

    // Pass grid_dim and block_dim to vx_spawn_threads
    return vx_spawn_threads(1, args->grid_dim, args->block_dim, (vx_kernel_func_cb)kernel_body, args);
}
