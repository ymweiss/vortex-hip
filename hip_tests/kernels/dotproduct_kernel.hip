// Dot product kernel extracted for Polygeist conversion
// Original: hip_tests/dotproduct.hip
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void dotproduct_kernel(const float* a, const float* b, float* partial_sums, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes one element's contribution
  float sum = 0.0f;
  if (tid < count) {
    sum = a[tid] * b[tid];
  }

  // Store partial result (reduction would happen in separate step)
  if (tid < count) {
    partial_sums[tid] = sum;
  }
}

void launch_dotproduct(const float* d_a, const float* d_b, float* d_partial, uint32_t count, int threads_per_block) {
  int num_blocks = (count + threads_per_block - 1) / threads_per_block;
  dotproduct_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_partial, count);
}
