#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE int
#endif

__global__ void sort_kernel(TYPE* src, TYPE* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  auto ref_value = src[idx];

  uint32_t pos = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur_value = src[i];
    pos += (cur_value < ref_value) || ((cur_value == ref_value) && (i < idx));
  }

  dst[pos] = ref_value;
}

void launch_sort(TYPE* src, TYPE* dst, uint32_t num_points, int num_blocks, int threads_per_block) {
  sort_kernel<<<num_blocks, threads_per_block>>>(src, dst, num_points);
}

