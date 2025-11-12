// Vortex kernel for CTA (Cooperative Thread Array) test
// Tests 3D grid and block dimensions
// Each thread computes: globalId + src[localId]

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
struct CtaArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    int32_t* src;
    int32_t* dst;
} __attribute__((packed));

// Kernel body - 3D indexing test
void kernel_body(CtaArgs* __UNIFORM__ args) {
    // Calculate CTA (block) size
    uint32_t cta_size = args->block_dim[0] * args->block_dim[1] * args->block_dim[2];

    // Calculate 3D block ID (flattened)
    uint32_t blockId = blockIdx.x +
                       blockIdx.y * args->grid_dim[0] +
                       blockIdx.z * args->grid_dim[0] * args->grid_dim[1];

    // Calculate 3D local thread ID (flattened)
    uint32_t localId = threadIdx.x +
                       threadIdx.y * args->block_dim[0] +
                       threadIdx.z * args->block_dim[0] * args->block_dim[1];

    // Calculate global thread ID
    uint32_t globalId = localId + blockId * cta_size;

    // Compute: dst[globalId] = globalId + src[localId]
    args->dst[globalId] = globalId + args->src[localId];
}

// Device main - spawns threads in 3D grid
int main() {
    CtaArgs* args = (CtaArgs*)csr_read(VX_CSR_MSCRATCH);

    // 3D grid for full indexing test
    return vx_spawn_threads(3, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
