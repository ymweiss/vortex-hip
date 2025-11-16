// SGEMM (Matrix Multiplication) Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/sgemm
// Computes C = A * B for square matrices

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
      exit(1) \
    } \
  } while (0)

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for matrix multiplication
__global__ void sgemm_kernel(TYPE* A, TYPE* B, TYPE* C, uint32_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < size && col < size) {
    TYPE sum = 0;
    for (int e = 0; e < size; ++e) {
      sum += A[row * size + e] * B[e * size + col];
    }
    C[row * size + col] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 4;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t num_elements = size * size;
  uint32_t buf_size = num_elements * sizeof(TYPE);

  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<TYPE> h_A(num_elements);
  std::vector<TYPE> h_B(num_elements);
  std::vector<TYPE> h_C(num_elements);

  // Initialize matrices with random values
  for (uint32_t i = 0; i < num_elements; ++i) {
    h_A[i] = static_cast<TYPE>(rand()) / RAND_MAX;
    h_B[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  TYPE *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, buf_size));
  HIP_CHECK(hipMalloc(&d_B, buf_size));
  HIP_CHECK(hipMalloc(&d_C, buf_size));

  // Upload matrices
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

  hipLaunchKernelGGL(sgemm_kernel, blocks, threads, 0, 0,
                     d_A, d_B, d_C, size);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result with CPU computation
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
      TYPE sum = 0;
      for (uint32_t e = 0; e < size; ++e) {
        sum += h_A[row * size + e] * h_B[e * size + col];
      }
      TYPE cur = h_C[row * size + col];
      TYPE diff = std::abs(cur - sum);

      if (diff > FLOAT_ULP * std::numeric_limits<TYPE>::epsilon() * size) {
        if (errors < 100) {
          printf("*** error: [%d,%d] expected=%f, actual=%f\n", row, col, sum, cur);
        }
        ++errors;
      }
    }
  }

  // Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
