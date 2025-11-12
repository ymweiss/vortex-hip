// HIP-style vector addition kernel for Vortex
// This demonstrates metadata generation for a real kernel

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure (will be populated by HIP runtime via metadata)
struct VecAddArgs {
    uint32_t grid_dim[3];    // Added by runtime
    uint32_t block_dim[3];   // Added by runtime
    uint64_t shared_mem;     // Added by runtime
    float*   a;              // Kernel argument 0
    float*   b;              // Kernel argument 1
    float*   c;              // Kernel argument 2
    uint32_t n;              // Kernel argument 3
} __attribute__((packed));

// Kernel body - executed by each thread
void kernel_body(VecAddArgs* __UNIFORM__ args) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < args->n) {
        args->c[tid] = args->a[tid] + args->b[tid];
    }
}

// Device main - spawns threads
int main() {
    VecAddArgs* args = (VecAddArgs*)csr_read(VX_CSR_MSCRATCH);

    // Calculate total threads needed
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];

    return vx_spawn_threads(1, &num_threads, nullptr, (vx_kernel_func_cb)kernel_body, args);
}
