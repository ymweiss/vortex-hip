// HIP host code for dot product test with shared memory
// Adapted from tests/dotproduct.cpp for Vortex HIP runtime

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
  uint32_t size = 16;

  // Parse command line
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

  // Configure kernel launch parameters
  const uint32_t threadsPerBlock = 8;  // Small for testing
  const uint32_t blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
  uint32_t dst_buf_size = blocksPerGrid * sizeof(TYPE);

  std::cout << "blocks per grid: " << blocksPerGrid << std::endl;
  std::cout << "threads per block: " << threadsPerBlock << std::endl;
  std::cout << "shared memory per block: " << (threadsPerBlock * sizeof(TYPE)) << " bytes" << std::endl;

  // Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_src0, *d_src1, *d_dst;
  HIP_CHECK(hipMalloc((void**)&d_src0, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_src1, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_dst, dst_buf_size));  // One result per block

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(blocksPerGrid);

  // Initialize data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = rand() % 10;  // Small values to avoid overflow
    h_src1[i] = rand() % 10;
  }

  // Upload source buffers
  std::cout << "upload source buffer0" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  std::cout << "upload source buffer1" << std::endl;
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel with shared memory
  std::cout << "launch kernel" << std::endl;
  size_t sharedMemBytes = threadsPerBlock * sizeof(TYPE);
  void* args[] = {&d_src0, &d_src1, &d_dst, &num_points};

  HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                            dim3(blocksPerGrid),
                            dim3(threadsPerBlock),
                            args,
                            sharedMemBytes,  // Shared memory size
                            nullptr));

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download result (partial sums from each block)
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // Final reduction on CPU (sum all block results)
  std::cout << "final reduction on CPU" << std::endl;
  TYPE gpu_result = 0;
  for (uint32_t i = 0; i < blocksPerGrid; ++i) {
    gpu_result += h_dst[i];
  }

  // Calculate reference result on CPU
  std::cout << "verify result" << std::endl;
  TYPE cpu_result = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    cpu_result += h_src0[i] * h_src1[i];
  }

  std::cout << "GPU result: " << gpu_result << std::endl;
  std::cout << "CPU result: " << cpu_result << std::endl;

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_src1));
  HIP_CHECK(hipFree(d_dst));

  if (gpu_result != cpu_result) {
    std::cout << "FAILED! Results don't match." << std::endl;
    std::cout << "Difference: " << (gpu_result - cpu_result) << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
