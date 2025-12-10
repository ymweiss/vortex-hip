// Basic Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/basic
// Tests basic memory copy functionality using GPU

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define NONCE 0xdeadbeef

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

inline uint32_t shuffle(int i, uint32_t value) {
  return (value << i) | (value & ((1 << i)-1));
}

// HIP kernel - simple memory copy
// Each thread copies one element
__global__ void basic_kernel(int32_t* src, int32_t* dst, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    dst[tid] = src[tid];
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t count = 1;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    }
  }

  // Get device properties for num_cores equivalent
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  uint32_t num_points = count;  // Simplified - in Vortex this is count * num_cores
  uint32_t buf_size = num_points * sizeof(int32_t);

  printf("number of points: %u\n", num_points);
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // Initialize source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  // Allocate device memory
  printf("allocate device memory\n");
  int32_t *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Upload source buffer
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(basic_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src, d_dst, num_points);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return errors;
  }

  printf("PASSED!\n");
  return 0;
}
