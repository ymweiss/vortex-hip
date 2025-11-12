#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

#define HIP_CHECK(cmd)                                            \
  do {                                                           \
    hipError_t error = cmd;                                      \
    if (error != hipSuccess) {                                   \
      fprintf(stderr, "error: '%s' returned %d (%s)!\n",        \
              #cmd, static_cast<int>(error),                    \
              hipGetErrorString(error));                         \
      exit(-1);                                                  \
    }                                                            \
  } while (false)

#define FLOAT_ULP 6

// Compute function used by both host and device
__host__ __device__ inline float madmax_compute(uint32_t row, uint32_t col, uint32_t size) {
  // Initialize 16 independent accumulators using thread indices
  float a0 = (row * size + col) * 0.5f;
  float a1 = (col * size + row) * 0.5f;
  float a2 = a0 + a1;
  float a3 = a0 - a1;
  float a4 = a2 * 0.5f;
  float a5 = a3 * 0.5f;
  float a6 = a4 + a5;
  float a7 = a4 - a5;
  float a8 = a6 * 0.5f;
  float a9 = a7 * 0.5f;
  float a10 = a8 + a9;
  float a11 = a8 - a9;
  float a12 = a10 * 0.5f;
  float a13 = a11 * 0.5f;
  float a14 = a12 + a13;
  float a15 = a12 - a13;

  // Perform massive independent FMADD chains (1024 iterations)
  for (int i = 0; i < 256; ++i) {
    a0 = a0 * a1 + a2;
    a1 = a1 * a2 + a3;
    a2 = a2 * a3 + a4;
    a3 = a3 * a4 + a5;
    a4 = a4 * a5 + a6;
    a5 = a5 * a6 + a7;
    a6 = a6 * a7 + a8;
    a7 = a7 * a8 + a9;
    a8 = a8 * a9 + a10;
    a9 = a9 * a10 + a11;
    a10 = a10 * a11 + a12;
    a11 = a11 * a12 + a13;
    a12 = a12 * a13 + a14;
    a13 = a13 * a14 + a15;
    a14 = a14 * a15 + a0;
    a15 = a15 * a0 + a1;
  }

  // Combine results to force dependency and write output
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// HIP Kernel
__global__ void kernel_madmax(float *dst, uint32_t size) {
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size && col < size) {
    dst[row * size + col] = madmax_compute(row, col, size);
  }
}

// Synthetic computation replica for verification (host version)
void compute_reference(float *ref, uint32_t size) {
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
      ref[row * size + col] = madmax_compute(row, col, size);
    }
  }
}

static void show_usage() {
  std::cout << "HIP Madmax Test." << std::endl;
  std::cout << "Usage: [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t &size) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t size = 32;

  // parse command arguments
  parse_args(argc, argv, size);

  uint32_t buf_size = size * size * sizeof(float);

  std::cout << "HIP Madmax Test" << std::endl;
  std::cout << "number of points: " << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  float *d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Setup grid and block dimensions
  // Use blocks of 16x16 for optimal occupancy on most GPUs
  dim3 blockDim(16, 16);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
               (size + blockDim.y - 1) / blockDim.y);

  std::cout << "Grid dimensions: " << gridDim.x << "x" << gridDim.y << std::endl;
  std::cout << "Block dimensions: " << blockDim.x << "x" << blockDim.y << std::endl;

  // launch kernel
  std::cout << "launch kernel" << std::endl;
  hipLaunchKernelGGL(kernel_madmax, gridDim, blockDim, 0, 0, d_dst, size);
  HIP_CHECK(hipGetLastError());

  // wait for kernel completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // download destination buffer
  std::vector<float> h_C(size * size);
  std::cout << "download result" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify results
  std::cout << "verify result" << std::endl;
  int errors = 0;
  std::vector<float> h_ref(size * size);
  compute_reference(h_ref.data(), size);

  for (uint32_t i = 0; i < h_ref.size(); ++i) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t actual, expected;
    actual.f = h_C[i];
    expected.f = h_ref[i];

    if (std::abs(actual.i - expected.i) > FLOAT_ULP) {
      if (errors < 3) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, expected.f, actual.f);
      }
      ++errors;
    }
  }

  // cleanup
  HIP_CHECK(hipFree(d_dst));

  if (errors == 0) {
    std::cout << "Test PASSED" << std::endl;
  } else {
    std::cout << "Test FAILED with " << errors << " errors" << std::endl;
  }

  return (errors != 0);
}
