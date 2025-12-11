// Bitonic Sort - HIP version
// Converted from Vortex regression test: vortex/tests/regression/sort
// Implementation uses bitonic sorting network for parallel GPU sorting

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#ifndef TYPE
#define TYPE int
#endif

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

// Bitonic sort kernel - performs one comparison-exchange step
__global__ void bitonic_sort_step(TYPE* data, uint32_t num_points, uint32_t stage, uint32_t step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_points) return;

  // Calculate partner index for comparison
  uint32_t ixj = idx ^ step;

  if (ixj > idx) {
    if (ixj < num_points) {
      TYPE a = data[idx];
      TYPE b = data[ixj];

      // Determine sort direction based on stage
      bool ascending = ((idx & stage) == 0);

      // Swap if needed
      if ((a > b) == ascending) {
        data[idx] = b;
        data[ixj] = a;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

void bitonic_sort_host(TYPE* h_data, uint32_t size) {
  // Host reference implementation
  for (uint32_t stage = 2; stage <= size; stage <<= 1) {
    for (uint32_t step = stage >> 1; step > 0; step >>= 1) {
      for (uint32_t i = 0; i < size; i++) {
        uint32_t ixj = i ^ step;
        if (ixj > i && ixj < size) {
          TYPE a = h_data[i];
          TYPE b = h_data[ixj];
          bool ascending = ((i & stage) == 0);
          if ((a > b) == ascending) {
            h_data[i] = b;
            h_data[ixj] = a;
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 64;  // Default size - must be power of 2 for bitonic sort

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  // Ensure size is power of 2 for bitonic sort
  uint32_t power = 1;
  while (power < size) power <<= 1;
  if (power != size) {
    printf("Warning: size must be power of 2 for bitonic sort, rounding up to %u\n", power);
    size = power;
  }

  std::srand(50);

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  printf("number of points: %u\n", num_points);
  std::cout << "data type: " << (sizeof(TYPE) == 4 ? "int" : "unknown") << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<TYPE> h_src(num_points);
  std::vector<TYPE> h_dst(num_points);
  std::vector<TYPE> h_ref(num_points);

  // Generate random data
  for (uint32_t i = 0; i < num_points; ++i) {
    auto r = static_cast<float>(std::rand()) / RAND_MAX;
    auto value = static_cast<TYPE>(r * num_points);
    h_src[i] = value;
  }

  // Generate reference result using host implementation
  h_ref = h_src;
  bitonic_sort_host(h_ref.data(), num_points);

  // Allocate device memory
  printf("allocate device memory\n");
  TYPE *d_data;
  HIP_CHECK(hipMalloc(&d_data, buf_size));

  // Upload source buffer
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_data, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch bitonic sort kernels
  printf("launch bitonic sort kernels\n");
  int threads_per_block = 256;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  // Bitonic sort stages
  for (uint32_t stage = 2; stage <= num_points; stage <<= 1) {
    for (uint32_t step = stage >> 1; step > 0; step >>= 1) {
      hipLaunchKernelGGL(bitonic_sort_step, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                         d_data, num_points, stage, step);
      HIP_CHECK(hipGetLastError());
    }
  }

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_dst.data(), d_data, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    TYPE ref = h_ref[i];
    TYPE cur = h_dst[i];
    if (cur != ref) {
      if (errors < 100) {
        std::cout << "error at result #" << i << ": actual=" << cur << ", expected=" << ref << std::endl;
      }
      ++errors;
    }
  }

  // Additional verification: check if sorted
  bool is_sorted = true;
  for (uint32_t i = 1; i < num_points; ++i) {
    if (h_dst[i] < h_dst[i-1]) {
      is_sorted = false;
      break;
    }
  }

  if (!is_sorted && errors == 0) {
    printf("Warning: array is not sorted but matches reference\n");
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_data));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
