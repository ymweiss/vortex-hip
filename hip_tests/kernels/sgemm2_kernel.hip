#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void sgemm2_kernel(TYPE* A, TYPE* B, TYPE* C, int size, int tile_size) {
  extern __shared__ TYPE shared_mem[];
  TYPE* local_A = shared_mem;
  TYPE* local_B = shared_mem + tile_size * tile_size;

  int g_row = blockIdx.x * blockDim.x + threadIdx.x;
  int g_col = blockIdx.y * blockDim.y + threadIdx.y;

  int l_row = threadIdx.x;
  int l_col = threadIdx.y;

  TYPE sum = 0;

  for (int k = 0; k < size; k += tile_size) {
    if (g_row < size && (k + l_col) < size) {
      local_A[l_row * tile_size + l_col] = A[g_row * size + (k + l_col)];
    } else {
      local_A[l_row * tile_size + l_col] = 0;
    }

    if ((k + l_row) < size && g_col < size) {
      local_B[l_row * tile_size + l_col] = B[(k + l_row) * size + g_col];
    } else {
      local_B[l_row * tile_size + l_col] = 0;
    }

    __syncthreads();

    for (int j = 0; j < tile_size; ++j) {
      sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
    }

    __syncthreads();
  }

  if (g_row < size && g_col < size) {
    C[g_row * size + g_col] = sum;
  }
}

void launch_sgemm2(TYPE* A, TYPE* B, TYPE* C, int size, int tile_size, int num_blocks, int threads_per_block) {
  sgemm2_kernel<<<num_blocks, threads_per_block>>>(A, B, C, size, tile_size);
}

