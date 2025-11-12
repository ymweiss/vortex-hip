// Vortex kernel for matrix multiplication (SGEMM)
// Adapted from tests/sgemm.cpp

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
// NOTE: Using 4-argument pattern (3 ptr + 1 uint32) to match metadata generator fallback
struct SgemmArgs {
    // Runtime-provided fields (filled by Vortex runtime)
    uint32_t grid_dim[3];      // Grid dimensions (x, y, z)
    uint32_t block_dim[3];     // Block dimensions (x, y, z)
    uint64_t shared_mem;       // Shared memory size (unused for this kernel)

    // User arguments
    int32_t* A;                // Matrix A (size x size)
    int32_t* B;                // Matrix B (size x size)
    int32_t* C;                // Matrix C (size x size) - output
    uint32_t size;             // Matrix dimension
} __attribute__((packed));

// Kernel body - performs matrix multiplication
// C[row][col] = sum(A[row][k] * B[k][col]) for k in [0, size)
void kernel_body(SgemmArgs* __UNIFORM__ args) {
    // Calculate 2D position in grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (col < args->size && row < args->size) {
        int32_t sum = 0;
        for (uint32_t e = 0; e < args->size; ++e) {
            sum += args->A[row * args->size + e] * args->B[e * args->size + col];
        }
        args->C[row * args->size + col] = sum;
    }
}

// Device main - spawns threads for 2D grid
int main() {
    SgemmArgs* args = (SgemmArgs*)csr_read(VX_CSR_MSCRATCH);

    // Calculate total threads needed for 2D grid
    uint32_t num_threads[2];
    num_threads[0] = args->grid_dim[0] * args->block_dim[0];  // X dimension
    num_threads[1] = args->grid_dim[1] * args->block_dim[1];  // Y dimension

    return vx_spawn_threads(2, num_threads, nullptr, (vx_kernel_func_cb)kernel_body, args);
}
