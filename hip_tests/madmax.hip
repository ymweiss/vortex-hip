// Madmax (Memory Access Pattern) - HIP version
// Converted from Vortex regression test: vortex/tests/regression/madmax

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define FLOAT_ULP 6

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

///////////////////////////////////////////////////////////////////////////////

// Device function for madmax computation
__device__ float madmax_compute(uint32_t row, uint32_t col, uint32_t size) {
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

///////////////////////////////////////////////////////////////////////////////

// HIP kernel
__global__ void madmax_kernel(float* dst, uint32_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size && col < size) {
    dst[row * size + col] = madmax_compute(row, col, size);
  }
}

///////////////////////////////////////////////////////////////////////////////

// Host-side computation replica for verification
void compute_reference(float *ref, uint32_t size) {
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
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
      ref[row * size + col] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 32;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  uint32_t num_points = size * size;
  uint32_t buf_size = num_points * sizeof(float);

  printf("number of points: %u\n", size);
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate device memory
  printf("allocate device memory\n");
  float *d_dst;
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Launch kernel
  printf("launch kernel\n");
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((size + threads_per_block.x - 1) / threads_per_block.x,
                  (size + threads_per_block.y - 1) / threads_per_block.y);

  hipLaunchKernelGGL(madmax_kernel, num_blocks, threads_per_block, 0, 0,
                     d_dst, size);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  std::vector<float> h_dst(num_points);
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  std::vector<float> h_ref(num_points);
  compute_reference(h_ref.data(), size);

  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t actual, expected;
    actual.f = h_dst[i];
    expected.f = h_ref[i];

    if (std::abs(actual.i - expected.i) > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, expected.f, actual.f);
      }
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
