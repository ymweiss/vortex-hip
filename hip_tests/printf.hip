// Printf - HIP version
// Converted from Vortex regression test: vortex/tests/regression/printf

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

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

// HIP kernel - prints core ID, task ID, and character value
__global__ void printf_kernel(char* src_ptr, uint32_t num_points) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    char value = 'A' + src_ptr[idx];
    printf("cid=%d: task=%d, value=%c\n", threadIdx.x, idx, value);
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t count = 4;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    }
  }

  if (count == 0) {
    count = 1;
  }

  // Get device properties to calculate total threads
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  uint32_t num_points = count * 64;  // Using a reasonable thread count
  uint32_t buf_size = num_points * sizeof(char);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffer
  std::cout << "allocate host buffers" << std::endl;
  std::vector<char> h_src(num_points);

  // Generate input data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = (char)i;
  }

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  char* d_src;
  HIP_CHECK(hipMalloc(&d_src, buf_size));

  // Upload source buffer
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(printf_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src, num_points);

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src));

  std::cout << "PASSED!" << std::endl;
  return 0;
}
