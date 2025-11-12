// Vortex kernel for madmax - computational stress test
// Adapted from tests/madmax.cpp
// Tests maximum ALU utilization with massive FMADD chains

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
struct MadmaxArgs {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;

    // User arguments
    float* dst;
    uint32_t size;
    uint32_t dummy1;  // Padding for 4-arg metadata
    uint32_t dummy2;
} __attribute__((packed));

// Compute function - massive independent FMADD chains
static inline float madmax_compute(uint32_t row, uint32_t col, uint32_t size) {
    // Initialize 16 independent accumulators
    float a0 = (row * size + col) * 0.5f;
    float a1 = (col * size + row) * 0.5f;
    float a2 = a0 + a1;
    float a3 = a0 - a1;
    float a4 = a2 * 0.5f;
    float a5 = a3 * 0.5f;
    float a6 = a4 + a5;
    float a7 = a4 - a5;
    float a8 = a6 * 0.5f;
    float a9 = a7 * 0.5f;
    float a10 = a8 + a9;
    float a11 = a8 - a9;
    float a12 = a10 * 0.5f;
    float a13 = a11 * 0.5f;
    float a14 = a12 + a13;
    float a15 = a12 - a13;

    // Perform massive independent FMADD chains (256 iterations)
    // Each iteration does 16 FMADD operations
    for (int i = 0; i < 256; ++i) {
        a0 = a0 * a1 + a2;
        a1 = a1 * a2 + a3;
        a2 = a2 * a3 + a4;
        a3 = a3 * a4 + a5;
        a4 = a4 * a5 + a6;
        a5 = a5 * a6 + a7;
        a6 = a6 * a7 + a8;
        a7 = a7 * a8 + a9;
        a8 = a8 * a9 + a10;
        a9 = a9 * a10 + a11;
        a10 = a10 * a11 + a12;
        a11 = a11 * a12 + a13;
        a12 = a12 * a13 + a14;
        a13 = a13 * a14 + a15;
        a14 = a14 * a15 + a0;
        a15 = a15 * a0 + a1;
    }

    // Combine all results to force dependency
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
           a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// Kernel body - 2D grid computation
void kernel_body(MadmaxArgs* __UNIFORM__ args) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t size = args->size;

    if (row < size && col < size) {
        args->dst[row * size + col] = madmax_compute(row, col, size);
    }
}

// Device main - spawns threads in 2D grid
int main() {
    MadmaxArgs* args = (MadmaxArgs*)csr_read(VX_CSR_MSCRATCH);

    // 2D grid for matrix-style computation
    return vx_spawn_threads(2, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
