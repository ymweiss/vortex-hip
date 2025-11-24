#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void sgemv_kernel(float* A, float* x, float* y, uint32_t M, uint32_t N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  float sum = 0.0f;
  for (int col = 0; col < N; col += 4) {
    float* a = &A[row * N + col];
    float* b = &x[col];
    sum += a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  }
  y[row] = sum;
}

void launch_sgemv(float* A, float* x, float* y, uint32_t M, uint32_t N, int num_blocks, int threads_per_block) {
  sgemv_kernel<<<num_blocks, threads_per_block>>>(A, x, y, M, N);
}

