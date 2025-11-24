#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void sgemm_kernel(TYPE* A, TYPE* B, TYPE* C, uint32_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size && col < size) {
    TYPE sum = 0;
    for (int e = 0; e < size; ++e) {
      sum += A[row * size + e] * B[e * size + col];
    }
    C[row * size + col] = sum;
  }
}

void launch_sgemm(TYPE* A, TYPE* B, TYPE* C, uint32_t size, int num_blocks, int threads_per_block) {
  sgemm_kernel<<<num_blocks, threads_per_block>>>(A, B, C, size);
}

