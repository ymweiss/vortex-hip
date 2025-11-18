// Vector add kernel extracted for Polygeist conversion
// Original: hip_tests/vecadd.hip
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void vecadd_kernel(const float* a, const float* b, float* c, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    c[tid] = a[tid] + b[tid];
  }
}

void launch_vecadd(const float* d_a, const float* d_b, float* d_c, uint32_t count, int threads_per_block) {
  int num_blocks = (count + threads_per_block - 1) / threads_per_block;
  vecadd_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, count);
}
