#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void stencil3d_kernel(TYPE* A, TYPE* B, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int dep = blockIdx.z * blockDim.z + threadIdx.z;

  if (col >= size || row >= size || dep >= size) {
    return;
  }

  TYPE sum = 0;
  int count = 0;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nz = dep + dz;
        int ny = row + dy;
        int nx = col + dx;

        if (nz < 0) nz = 0;
        else if (nz >= size) nz = size - 1;
        if (ny < 0) ny = 0;
        else if (ny >= size) ny = size - 1;
        if (nx < 0) nx = 0;
        else if (nx >= size) nx = size - 1;

        sum += A[nz * size * size + ny * size + nx];
        count++;
      }
    }
  }

  B[dep * size * size + row * size + col] = sum / count;
}

void launch_stencil3d(TYPE* A, TYPE* B, int size, int num_blocks, int threads_per_block) {
  stencil3d_kernel<<<num_blocks, threads_per_block>>>(A, B, size);
}

