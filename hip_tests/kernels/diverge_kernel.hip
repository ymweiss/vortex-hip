#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

struct key_t {
  uint32_t user = 0;
};

__device__ inline int32_t min(int32_t a, int32_t b) { return a < b ? a : b; }
__device__ inline int32_t max(int32_t a, int32_t b) { return a > b ? a : b; }

__device__ __noinline__ void hacker(key_t* key, uint32_t task_id) {
  key->user = task_id;
}

__global__ void diverge_kernel(int32_t* src, int32_t* dst, uint32_t num_points, uint32_t samples) {
  uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (task_id >= num_points) return;

  int value = src[task_id];

  key_t key;
  uint32_t sample_count = samples;
  while (sample_count--) {
    hacker(&key, task_id);
    if ((key.user & 0x1) == 0) {
      value += 1;
    }
  }

  if (task_id >= 0x7fffffff) {
    value = 0;
  } else {
    value += 2;
  }

  if (task_id > 1) {
    if (task_id > 2) {
      value += 6;
    } else {
      value += 5;
    }
  } else {
    if (task_id > 0) {
      value += 4;
    } else {
      value += 3;
    }
  }

  if (task_id >= 0) {
    value += 7;
  } else {
    value = 0;
  }

  for (int i = 0, n = task_id; i < n; ++i) {
    value += src[i];
  }

  switch (task_id) {
  case 0:
    value += 1;
    break;
  case 1:
    value -= 1;
    break;
  case 2:
    value *= 3;
    break;
  case 3:
    value *= 5;
    break;
  default:
    break;
  }

  value += (task_id >= 0) ? ((task_id > 5) ? src[0] : task_id) : ((task_id < 5) ? src[1] : -task_id);

  value += min(src[task_id], value);
  value += max(src[task_id], value);

  dst[task_id] = value;
}

void launch_diverge(int32_t* src, int32_t* dst, uint32_t num_points, uint32_t samples, int num_blocks, int threads_per_block) {
  diverge_kernel<<<num_blocks, threads_per_block>>>(src, dst, num_points, samples);
}

