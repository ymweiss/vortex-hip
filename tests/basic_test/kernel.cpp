// Vortex kernel for basic test - memory copy operation
// Adapted from tests/basic.cpp

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
// NOTE: Metadata generator fallback expects 4 args (3 ptrs + 1 int)
// So we add a dummy 4th argument to match the fallback pattern
struct BasicArgs {
    uint32_t grid_dim[3];    // Added by runtime
    uint32_t block_dim[3];   // Added by runtime
    uint64_t shared_mem;     // Added by runtime
    int32_t* src;            // Kernel argument 0
    int32_t* dst;            // Kernel argument 1
    int32_t* dummy;          // Kernel argument 2 (unused - for metadata compat)
    uint32_t count;          // Kernel argument 3 (total elements, not per-block)
} __attribute__((packed));

// Kernel body - copy data from src to dst
void kernel_body(BasicArgs* __UNIFORM__ args) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread copies one element
    if (idx < args->count) {
        args->dst[idx] = args->src[idx];
    }
}

// Device main - spawns threads
int main() {
    BasicArgs* args = (BasicArgs*)csr_read(VX_CSR_MSCRATCH);

    // Calculate total threads needed
    uint32_t num_threads = args->grid_dim[0] * args->block_dim[0];

    return vx_spawn_threads(1, &num_threads, nullptr, (vx_kernel_func_cb)kernel_body, args);
}
