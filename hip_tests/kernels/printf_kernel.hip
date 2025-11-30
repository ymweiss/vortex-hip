#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void printf_kernel(char* src, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  char value = 'A' + src[idx];
  printf("task=%d, value=%c\n", idx, value);
}

void launch_printf(char* src, uint32_t num_points, int num_blocks, int threads_per_block) {
  printf_kernel<<<num_blocks, threads_per_block>>>(src, num_points);
}

