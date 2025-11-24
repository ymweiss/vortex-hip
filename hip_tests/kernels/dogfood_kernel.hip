// Dogfood kernel - comprehensive operation test
// Extracted from Vortex regression test: vortex/tests/regression/dogfood
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE int32_t
#endif

// Simple arithmetic test kernel
__global__ void dogfood_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points, uint32_t testid) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_points) return;
  
  TYPE a = src0[tid];
  TYPE b = src1[tid];
  TYPE result;
  
  // Simplified test selection (original has 24 tests)
  switch(testid) {
    case 0: result = a + b; break;           // iadd
    case 1: result = a * b; break;           // imul
    case 2: result = (b != 0) ? a / b : 0; break; // idiv
    case 3: result = a - b; break;           // isub
    default: result = a + b; break;
  }
  
  dst[tid] = result;
}

void launch_dogfood(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points, uint32_t testid, int num_blocks, int threads_per_block) {
  dogfood_kernel<<<num_blocks, threads_per_block>>>(src0, src1, dst, num_points, testid);
}
