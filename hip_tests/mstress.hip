// Memory Stress Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/mstress
// Tests indirect memory access patterns with multiple loads per thread

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>

#define NUM_LOADS 8

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

union Float_t {
  float f;
  int32_t i;
  struct {
    uint32_t man  : 23;
    uint32_t exp  : 8;
    uint32_t sign : 1;
  } parts;
};

inline bool almost_equal_ulp(float a, float b, int32_t ulp = 6) {
  Float_t fa{a}, fb{b};
  auto d = std::abs(fa.i - fb.i);
  if (d > ulp) {
    return false;
  }
  return true;
}

inline bool almost_equal(float a, float b) {
  if (a == b)
    return true;
  return almost_equal_ulp(a, b);
}

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for memory stress test
// Performs NUM_LOADS indirect memory accesses per thread and multiplies values
__global__ void mstress_kernel(uint32_t* addr_ptr, float* src_ptr, float* dst_ptr,
                               uint32_t stride) {
  uint32_t offset = blockIdx.x * stride;
  uint32_t tid = threadIdx.x;

  if (tid < stride) {
    float value = 0.0f;
    for (uint32_t j = 0; j < NUM_LOADS; ++j) {
      uint32_t addr = offset + tid + j;
      uint32_t index = addr_ptr[addr];
      value *= src_ptr[index];
    }
    dst_ptr[offset + tid] = value;
  }
}

///////////////////////////////////////////////////////////////////////////////

void gen_src_data(std::vector<float>& test_data,
                  std::vector<uint32_t>& addr_table,
                  uint32_t num_points,
                  uint32_t num_addrs) {
  test_data.resize(num_points);
  addr_table.resize(num_addrs);

  for (uint32_t i = 0; i < num_points; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    test_data[i] = r;
  }

  for (uint32_t i = 0; i < num_addrs; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    uint32_t index = static_cast<uint32_t>(r * num_points);
    if (index >= num_points) index = num_points - 1;
    addr_table[i] = index;
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

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // Get device properties to calculate thread configuration
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  // Calculate total threads based on device capability
  // Using a simplified approach: use warpSize as thread count per block
  uint32_t threads_per_block = deviceProp.warpSize;
  uint32_t num_blocks = deviceProp.multiProcessorCount;
  uint32_t total_threads = num_blocks * threads_per_block;

  uint32_t stride = count;
  uint32_t num_points = count * total_threads;
  uint32_t num_addrs = num_points + NUM_LOADS - 1;

  uint32_t addr_buf_size = num_addrs * sizeof(uint32_t);
  uint32_t src_buf_size  = num_points * sizeof(float);
  uint32_t dst_buf_size  = num_points * sizeof(float);

  printf("number of points: %u\n", num_points);
  std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;
  std::cout << "src buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "dst buffer size: " << dst_buf_size << " bytes" << std::endl;
  printf("num_blocks: %u\n", num_blocks);
  printf("threads_per_block: %u\n", threads_per_block);
  printf("stride: %u\n", stride);

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<uint32_t> h_addr;
  std::vector<float> h_src;
  std::vector<float> h_dst(num_points);
  gen_src_data(h_src, h_addr, num_points, num_addrs);

  // Allocate device memory
  printf("allocate device memory\n");
  uint32_t *d_addr;
  float *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_addr, addr_buf_size));
  HIP_CHECK(hipMalloc(&d_src, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // Upload address buffer
  printf("upload address buffer\n");
  HIP_CHECK(hipMemcpy(d_addr, h_addr.data(), addr_buf_size, hipMemcpyHostToDevice));

  // Upload source buffer
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  hipLaunchKernelGGL(mstress_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_addr, d_src, d_dst, stride);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    float ref = 0.0f;
    for (uint32_t j = 0; j < NUM_LOADS; ++j) {
      uint32_t addr = i + j;
      uint32_t index = h_addr[addr];
      float value = h_src[index];
      ref *= value;
    }

    float cur = h_dst[i];
    if (!almost_equal(cur, ref)) {
      if (errors < 100) {
        std::cout << "error at result #" << i
                  << ": actual " << cur << ", expected " << ref << std::endl;
      }
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_addr));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
