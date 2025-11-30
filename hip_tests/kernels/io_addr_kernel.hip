// I/O Address kernel - indirect memory access test
// Extracted from Vortex regression test: vortex/tests/regression/io_addr
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

__global__ void io_addr_kernel(uint64_t* src, uint32_t* dst, uint32_t num_points) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_points) {
    // Indirect memory access: read address from src, then read from that address
    uint64_t addr = src[tid];
    int32_t* ptr = (int32_t*)addr;
    dst[tid] = (uint32_t)(*ptr);
  }
}

void launch_io_addr(uint64_t* src, uint32_t* dst, uint32_t num_points, int num_blocks, int threads_per_block) {
  io_addr_kernel<<<num_blocks, threads_per_block>>>(src, dst, num_points);
}
