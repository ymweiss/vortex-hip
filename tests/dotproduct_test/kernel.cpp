// Vortex kernel for dot product with reduction
// Adapted from tests/dotproduct.cpp
// Uses shared memory for block-level reduction

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
struct DotproductArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;        // Shared memory size in bytes

    // User arguments
    int32_t* src0;
    int32_t* src1;
    int32_t* dst;               // Output: one result per block
    uint32_t num_points;
} __attribute__((packed));

// Kernel body - dot product with reduction using shared memory
void kernel_body(DotproductArgs* __UNIFORM__ args) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    uint32_t blockDim_x = args->block_dim[0];

    // Allocate shared memory for this block using Vortex's __local_mem macro
    // The size is passed via args->shared_mem from the runtime
    auto cache = reinterpret_cast<int32_t*>(__local_mem(args->shared_mem));

    // Calculate partial dot product for this thread
    int32_t temp = 0;
    while (tid < args->num_points) {
        temp += args->src0[tid] * args->src1[tid];
        tid += blockDim_x * args->grid_dim[0];
    }

    // Store in shared memory
    cache[cacheIndex] = temp;

    // Synchronize threads in this block
    __syncthreads();

    // Reduction in shared memory
    // Each iteration halves the number of active threads
    int i = blockDim_x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Thread 0 writes block result
    if (cacheIndex == 0) {
        args->dst[blockIdx.x] = cache[0];
    }
}

// Device main - spawns threads
int main() {
    DotproductArgs* args = (DotproductArgs*)csr_read(VX_CSR_MSCRATCH);

    // Pass grid_dim and block_dim to vx_spawn_threads
    return vx_spawn_threads(1, args->grid_dim, args->block_dim, (vx_kernel_func_cb)kernel_body, args);
}
