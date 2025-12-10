// Simple shared memory test for Vortex
// Tests static shared memory lowering

#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32

// Kernel using static shared memory
__global__ void reduce_kernel(int* input, int* output, int n) {
  // Static shared memory declaration
  __shared__ int cache[BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (gid < n) {
    cache[tid] = input[gid];
  } else {
    cache[tid] = 0;
  }

  __syncthreads();

  // Simple reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      cache[tid] += cache[tid + stride];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[blockIdx.x] = cache[0];
  }
}
