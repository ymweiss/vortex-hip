#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE int32_t
#endif

__global__ void fence_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t task_size, uint32_t num_tasks) {
  uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (task_id >= num_tasks) return;

  uint32_t offset = task_id * task_size;
  for (uint32_t i = 0; i < task_size; ++i) {
    dst[offset + i] = src0[offset + i] + src1[offset + i];
  }

  __syncthreads();
}

void launch_fence(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t task_size, uint32_t num_tasks, int num_blocks, int threads_per_block) {
  fence_kernel<<<num_blocks, threads_per_block>>>(src0, src1, dst, task_size, num_tasks);
}

