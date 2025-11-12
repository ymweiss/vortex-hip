// HIP host code for ReLU activation test
// Adapted from tests/relu.cpp for Vortex HIP runtime

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
  std::cout << "data type: integer" << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Initialize with mix of positive and negative values
  for (uint32_t i = 0; i < num_points; ++i) {
    // Generate values in range [-500, 500]
    h_src0[i] = (rand() % 1000) - 500;
  }

  // Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_src0, *d_dst;
  HIP_CHECK(hipMalloc((void**)&d_src0, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

  // Copy data to device
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  int blockSize = 256;
  int numBlocks = (num_points + blockSize - 1) / blockSize;

  std::cout << "  grid: " << numBlocks << " blocks" << std::endl;
  std::cout << "  block: " << blockSize << " threads" << std::endl;
  std::cout << "  total threads: " << (numBlocks * blockSize) << std::endl;

  // NOTE: Add dummy pointer to match 4-argument fallback pattern
  TYPE* dummy_ptr = nullptr;
  void* args[] = {&d_src0, &d_dst, &dummy_ptr, &num_points};
  HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                            dim3(numBlocks),
                            dim3(blockSize),
                            args,
                            0,        // shared memory bytes
                            nullptr   // stream
                            ));

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    TYPE ref = (h_src0[i] < 0) ? 0 : h_src0[i];
    TYPE cur = h_dst[i];
    if (cur != ref) {
      if (errors < 10) {
        std::cout << "*** error: [" << i << "] expected=" << ref
                  << ", actual=" << cur
                  << " (src=" << h_src0[i] << ")"
                  << std::endl;
      }
      ++errors;
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
