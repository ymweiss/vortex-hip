// HIP host code for SGEMM (matrix multiplication) test
// Adapted from tests/sgemm.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

// Using int32_t for this test (matches kernel)
typedef int32_t TYPE;

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

// Kernel function handle (set by registration)
extern void* kernel_body_handle;

// CPU reference implementation
static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum = 0;
      for (uint32_t e = 0; e < width; ++e) {
          sum += A[row * width + e] * B[e * width + col];
      }
      out[row * width + col] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t size = 8;  // Start with small size for testing

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t size_sq = size * size;
  uint32_t buf_size = size_sq * sizeof(TYPE);

  std::cout << "data type: integer" << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Generate source data
  std::cout << "generate source data" << std::endl;
  std::vector<TYPE> h_A(size_sq);
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C(size_sq);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = rand() % 10;  // Small values to avoid overflow
    h_B[i] = rand() % 10;
  }

  // Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc((void**)&d_A, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_B, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_C, buf_size));

  // Upload matrix A buffer
  std::cout << "upload matrix A buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));

  // Upload matrix B buffer
  std::cout << "upload matrix B buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel with 2D grid
  std::cout << "launch kernel" << std::endl;

  // Use smaller block size for testing
  dim3 blockSize(4, 4);  // 16 threads per block
  dim3 numBlocks((size + blockSize.x - 1) / blockSize.x,
                 (size + blockSize.y - 1) / blockSize.y);

  std::cout << "  grid: " << numBlocks.x << "x" << numBlocks.y << " blocks" << std::endl;
  std::cout << "  block: " << blockSize.x << "x" << blockSize.y << " threads" << std::endl;
  std::cout << "  total threads: " << (numBlocks.x * numBlocks.y * blockSize.x * blockSize.y) << std::endl;

  void* args[] = {&d_A, &d_B, &d_C, &size};
  HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                            numBlocks,
                            blockSize,
                            args,
                            0,        // shared memory bytes
                            nullptr   // stream
                            ));

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (h_C[i] != h_ref[i]) {
        if (errors < 10) {
          uint32_t row = i / size;
          uint32_t col = i % size;
          std::cout << "*** error: [" << row << "][" << col << "] expected=" << h_ref[i]
                    << ", actual=" << h_C[i] << std::endl;
        }
        ++errors;
      }
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
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
