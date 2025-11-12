// Vortex kernel for mstress - memory stress test
// Adapted from tests/mstress.cpp
// Tests memory subsystem with indirect addressing and multiple loads

#include <vx_spawn.h>
#include <stdint.h>

#define NUM_LOADS 8

// Kernel argument structure
struct MstressArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    uint32_t* addr;
    float* src;
    float* dst;
    uint32_t stride;
} __attribute__((packed));

// Kernel body - memory stress with indirect addressing
void kernel_body(MstressArgs* __UNIFORM__ args) {
    uint32_t blockIdx_x = blockIdx.x;
    uint32_t offset = blockIdx_x * args->stride;

    // Each block processes 'stride' elements
    for (uint32_t i = 0; i < args->stride; ++i) {
        float value = 0.0f;

        // Multiple loads with indirect addressing
        for (uint32_t j = 0; j < NUM_LOADS; ++j) {
            uint32_t addr_idx = offset + i + j;
            uint32_t index = args->addr[addr_idx];
            value *= args->src[index];
        }

        args->dst[offset + i] = value;
    }
}

// Device main - spawns threads
int main() {
    MstressArgs* args = (MstressArgs*)csr_read(VX_CSR_MSCRATCH);

    // 1D grid, each block processes independently
    return vx_spawn_threads(1, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
