// Vortex kernel for ReLU activation function
// Adapted from tests/relu.cpp

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
// NOTE: ReLU naturally needs only 3 args (2 ptr + 1 uint32)
// But we add dummy pointer to match metadata generator fallback (4 args total)
struct ReluArgs {
    // Runtime-provided fields (filled by Vortex runtime)
    uint32_t grid_dim[3];      // Grid dimensions
    uint32_t block_dim[3];     // Block dimensions
    uint64_t shared_mem;       // Shared memory size (unused for this kernel)

    // User arguments
    int32_t* src0;             // Source buffer
    int32_t* dst;              // Destination buffer
    int32_t* dummy;            // Dummy pointer (for metadata compatibility)
    uint32_t num_points;       // Number of elements to process
} __attribute__((packed));

// Kernel body - performs ReLU activation: dst[i] = max(0, src[i])
void kernel_body(ReluArgs* __UNIFORM__ args) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check and perform ReLU
    if (idx < args->num_points) {
        int32_t value = args->src0[idx];
        args->dst[idx] = (value < 0) ? 0 : value;
    }
}

// Device main - spawns threads
int main() {
    ReluArgs* args = (ReluArgs*)csr_read(VX_CSR_MSCRATCH);

    // Calculate total threads needed
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];

    return vx_spawn_threads(1, &num_threads, nullptr, (vx_kernel_func_cb)kernel_body, args);
}
