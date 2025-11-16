// ReLU Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/relu
// Applies ReLU activation function: f(x) = max(0, x)

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#ifndef TYPE
#define TYPE float
#endif

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

// HIP kernel for ReLU activation
__global__ void relu_kernel(TYPE* src, TYPE* dst, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    TYPE val = src[idx];
    dst[idx] = (val > 0) ? val : 0;
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 16;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<TYPE> h_src(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Initialize with random values (some negative, some positive)
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = (static_cast<TYPE>(rand()) / RAND_MAX) * 2.0f - 1.0f; // Range: [-1, 1]
  }

  // Allocate device memory
  TYPE *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Upload source buffer
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(relu_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src, d_dst, num_points);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    TYPE ref = (h_src[i] > 0) ? h_src[i] : 0;
    TYPE cur = h_dst[i];
    TYPE diff = std::abs(cur - ref);

    if (diff > FLOAT_ULP * std::numeric_limits<TYPE>::epsilon()) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, ref, cur);
      }
      ++errors;
    }
  }

  // Cleanup
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
