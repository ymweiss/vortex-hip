#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void relu_kernel(TYPE* src, TYPE* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    TYPE val = src[idx];
    dst[idx] = (val > 0) ? val : 0;
  }
}

void launch_relu(TYPE* src, TYPE* dst, uint32_t num_points, int num_blocks, int threads_per_block) {
  relu_kernel<<<num_blocks, threads_per_block>>>(src, dst, num_points);
}

