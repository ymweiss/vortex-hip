#include "cuda.h"
#include "__clang_cuda_builtin_vars.h"

__global__ void simple_add(int* data, int value) {
    int idx = threadIdx.x;
    data[idx] += value;
}

void launch_kernel(int* data, int value, int num_threads) {
    simple_add<<<1, num_threads>>>(data, value);
}
