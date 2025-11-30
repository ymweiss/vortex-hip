#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void conv3_kernel(TYPE* I, TYPE* W, TYPE* O, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= width) {
    return;
  }

  int paddedWidth = width + 2;
  int paddedX = col + 1;
  int paddedY = row + 1;

  float sum = 0.0f;

  sum += I[(paddedY - 1) * paddedWidth + (paddedX - 1)] * W[0];
  sum += I[(paddedY - 1) * paddedWidth + paddedX] * W[1];
  sum += I[(paddedY - 1) * paddedWidth + (paddedX + 1)] * W[2];

  sum += I[paddedY * paddedWidth + (paddedX - 1)] * W[3];
  sum += I[paddedY * paddedWidth + paddedX] * W[4];
  sum += I[paddedY * paddedWidth + (paddedX + 1)] * W[5];

  sum += I[(paddedY + 1) * paddedWidth + (paddedX - 1)] * W[6];
  sum += I[(paddedY + 1) * paddedWidth + paddedX] * W[7];
  sum += I[(paddedY + 1) * paddedWidth + (paddedX + 1)] * W[8];

  O[row * width + col] = sum;
}

void launch_conv3(TYPE* I, TYPE* W, TYPE* O, int width, int num_blocks, int threads_per_block) {
  conv3_kernel<<<num_blocks, threads_per_block>>>(I, W, O, width);
}

