#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void cta_kernel(int* src, int* dst) {
  uint32_t cta_size = blockDim.x * blockDim.y * blockDim.z;
  uint32_t blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  uint32_t localId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  uint32_t globalId = localId + blockId * cta_size;
  dst[globalId] = globalId + src[localId];
}

void launch_cta(int* src, int* dst, int num_blocks, int threads_per_block) {
  cta_kernel<<<num_blocks, threads_per_block>>>(src, dst);
}

