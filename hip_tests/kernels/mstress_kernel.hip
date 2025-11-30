// Memory stress test kernel - multiple indirect loads
// Extracted from Vortex regression test: vortex/tests/regression/mstress
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef NUM_LOADS
#define NUM_LOADS 8
#endif

#ifndef TYPE
#define TYPE int32_t
#endif

__global__ void mstress_kernel(uint32_t* src0_addr, TYPE* src1_addr, TYPE* dst_addr, uint32_t stride) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t offset = tid * stride;
  
  TYPE result = 1;
  for (uint32_t i = 0; i < stride; ++i) {
    for (uint32_t j = 0; j < NUM_LOADS; ++j) {
      uint32_t index = src0_addr[offset + i + j];
      TYPE value = src1_addr[index];
      result *= value;
    }
  }
  
  dst_addr[tid] = result;
}

void launch_mstress(uint32_t* src0_addr, TYPE* src1_addr, TYPE* dst_addr, uint32_t stride, int num_blocks, int threads_per_block) {
  mstress_kernel<<<num_blocks, threads_per_block>>>(src0_addr, src1_addr, dst_addr, stride);
}
