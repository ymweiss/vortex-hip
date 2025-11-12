// HIP host code for fence test
// Adapted from tests/fence.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

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

extern void* kernel_body_handle;

int main(int argc, char *argv[]) {
  uint32_t count = 1;  // elements per block

  // Parse command line
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    }
  }

  // Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Get device properties
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  // Use modest values for testing
  uint32_t num_blocks = 4;  // Simple multi-block test
  uint32_t num_points = count * num_blocks;
  size_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of blocks: " << num_blocks << std::endl;
  std::cout << "elements per block: " << count << std::endl;
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Initialize data
  std::srand(50);
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = rand() % 100;
    h_src1[i] = rand() % 100;
  }

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_src0, *d_src1, *d_dst;
  HIP_CHECK(hipMalloc((void**)&d_src0, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_src1, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_dst, buf_size));

  // Copy data to device
  std::cout << "upload source buffers" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  void* args[] = {&d_src0, &d_src1, &d_dst, &count};
  HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                            dim3(num_blocks),
                            dim3(1),  // 1 thread per block for this test
                            args,
                            0,
                            nullptr));

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
    TYPE ref = h_src0[i] + h_src1[i];
    TYPE cur = h_dst[i];
    if (cur != ref) {
      if (errors < 10) {
        std::cout << "*** error: [" << i << "] expected=" << ref
                  << ", actual=" << cur << std::endl;
      }
      ++errors;
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_src1));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
