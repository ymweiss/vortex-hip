// Vortex kernel for vector addition
// Adapted from tests/vecadd.cpp

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
// NOTE: Using 4-argument pattern (3 ptr + 1 uint32) to match metadata generator fallback
struct VecaddArgs {
    // Runtime-provided fields (filled by Vortex runtime)
    uint32_t grid_dim[3];      // Grid dimensions
    uint32_t block_dim[3];     // Block dimensions
    uint64_t shared_mem;       // Shared memory size (unused for this kernel)

    // User arguments
    int32_t* src0;             // First source buffer
    int32_t* src1;             // Second source buffer
    int32_t* dst;              // Destination buffer
    uint32_t num_points;       // Number of elements to process
} __attribute__((packed));

// Kernel body - performs vector addition
void kernel_body(VecaddArgs* __UNIFORM__ args) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check and perform vector addition
    if (idx < args->num_points) {
        args->dst[idx] = args->src0[idx] + args->src1[idx];
    }
}

// Device main - spawns threads
int main() {
    VecaddArgs* args = (VecaddArgs*)csr_read(VX_CSR_MSCRATCH);

    // Calculate total threads needed
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];

    return vx_spawn_threads(1, &num_threads, nullptr, (vx_kernel_func_cb)kernel_body, args);
}
