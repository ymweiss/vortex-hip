// Vortex kernel for tiled matrix multiplication (SGEMM2)
// Adapted from tests/sgemm2.cpp
// Uses shared memory for tiling optimization

#include <vx_spawn.h>
#include <stdint.h>

// Kernel argument structure
struct Sgemm2Args {
    // Runtime-provided fields
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;        // Shared memory size in bytes

    // User arguments
    float* A;
    float* B;
    float* C;
    uint32_t size;              // Matrix dimension (size x size)
} __attribute__((packed));

// Kernel body - tiled matrix multiplication using shared memory
void kernel_body(Sgemm2Args* __UNIFORM__ args) {
    // Allocate shared memory for tiles of A and B
    // We need 2 tiles: one for A, one for B
    auto local_ptr = __local_mem(args->shared_mem);
    auto local_A = reinterpret_cast<float*>(local_ptr);
    auto local_B = local_A + args->block_dim[0] * args->block_dim[1];

    uint32_t size = args->size;
    // Tile size is the same as block dimension
    uint32_t tile_size = args->block_dim[0];

    // Determine global row and column indices
    uint32_t g_row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t g_col = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine local row and column indices within tile
    uint32_t l_row = threadIdx.x;
    uint32_t l_col = threadIdx.y;

    float sum = 0.0f;

    // Loop over tiles along the K dimension
    for (uint32_t k = 0; k < size; k += tile_size) {
        // Load tile of matrix A from global to shared memory
        // A tile: rows [g_row], columns [k..k+tile_size)
        local_A[l_row * tile_size + l_col] = args->A[g_row * size + (k + l_col)];

        // Load tile of matrix B from global to shared memory
        // B tile: rows [k..k+tile_size), columns [g_col]
        local_B[l_row * tile_size + l_col] = args->B[(k + l_row) * size + g_col];

        // Synchronize all threads in block before computing
        __syncthreads();

        // Compute partial sum using tiles in shared memory
        for (uint32_t j = 0; j < tile_size; ++j) {
            sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Store the computed sum into the result matrix C
    args->C[g_row * size + g_col] = sum;
}

// Device main - spawns threads in 2D grid
int main() {
    Sgemm2Args* args = (Sgemm2Args*)csr_read(VX_CSR_MSCRATCH);

    // 2D grid for matrix operations
    return vx_spawn_threads(2, args->grid_dim, args->block_dim, (vx_kernel_func_cb)kernel_body, args);
}
