// Vortex kernel for demo - vector addition
// Adapted from tests/demo.cpp
// Simple vector addition to demonstrate HIP runtime basics

#include <vx_spawn.h>
#include <stdint.h>

// Type definition
typedef int32_t TYPE;

// Kernel argument structure
struct DemoArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    TYPE* src0;
    TYPE* src1;
    TYPE* dst;
    uint32_t count;  // Elements per block
} __attribute__((packed));

// Kernel body - vector addition with thread distribution
void kernel_body(DemoArgs* __UNIFORM__ args) {
    uint32_t block_idx = blockIdx.x;
    uint32_t thread_idx = threadIdx.x;

    // Each block processes 'count' elements
    uint32_t offset = block_idx * args->count;

    // Distribute iterations across all threads in the block
    for (uint32_t i = thread_idx; i < args->count; i += args->block_dim[0]) {
        args->dst[offset + i] = args->src0[offset + i] + args->src1[offset + i];
    }
}

// Device main - spawns threads
int main() {
    DemoArgs* args = (DemoArgs*)csr_read(VX_CSR_MSCRATCH);

    // 1D grid with multiple threads per block
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
