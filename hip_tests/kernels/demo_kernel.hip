// Demo kernel - simple element-wise addition
// Extracted from Vortex regression test: vortex/tests/regression/demo
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE int32_t
#endif

__global__ void demo_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_points) {
    dst[tid] = src0[tid] + src1[tid];
  }
}

void launch_demo(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points, int num_blocks, int threads_per_block) {
  demo_kernel<<<num_blocks, threads_per_block>>>(src0, src1, dst, num_points);
}
