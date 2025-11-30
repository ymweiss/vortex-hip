#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__device__ __host__ unsigned int WangHash(unsigned int s) {
  s = (s ^ 61) ^ (s >> 16);
  s *= 9;
  s = s ^ (s >> 4);
  s *= 0x27d4eb2d;
  s = s ^ (s >> 15);
  return s;
}

__device__ __host__ unsigned int RandomInt(unsigned int s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

__device__ __host__ float RandomFloat(unsigned int s) {
  return RandomInt(s) * 2.3283064365387e-10f;
}

__global__ void dropout_kernel(TYPE* src0, TYPE* dst, float dropout_p, float multiplier, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  float rand_value = RandomFloat(WangHash(idx));
  TYPE scaled_value = src0[idx] * multiplier;
  dst[idx] = (rand_value < dropout_p) ? 0.0 : scaled_value;
}

void launch_dropout(TYPE* src0, TYPE* dst, float dropout_p, float multiplier, uint32_t num_points, int num_blocks, int threads_per_block) {
  dropout_kernel<<<num_blocks, threads_per_block>>>(src0, dst, dropout_p, multiplier, num_points);
}

