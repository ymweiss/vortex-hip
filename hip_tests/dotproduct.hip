// Dot Product Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/dotproduct
// Computes dot product using parallel reduction with shared memory

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

// HIP kernel for dot product with parallel reduction
__global__ void dotproduct_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points) {
  extern __shared__ TYPE cache[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  TYPE temp = 0;
  while (tid < num_points) {
    temp += src0[tid] * src1[tid];
    tid += blockDim.x * gridDim.x;
  }

  // Set the cache values
  cache[cacheIndex] = temp;

  __syncthreads();

  // Parallel reduction in shared memory
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0)
    dst[blockIdx.x] = cache[0];
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
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = static_cast<TYPE>(rand()) / RAND_MAX;
    h_src1[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  TYPE *d_src0, *d_src1, *d_dst;
  int threads_per_block = 64;
  int num_blocks = 4;

  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, num_blocks * sizeof(TYPE)));

  // Upload source buffers
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel with shared memory
  size_t shared_mem_size = threads_per_block * sizeof(TYPE);
  hipLaunchKernelGGL(dotproduct_kernel, dim3(num_blocks), dim3(threads_per_block),
                     shared_mem_size, 0,
                     d_src0, d_src1, d_dst, num_points);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download partial results
  std::vector<TYPE> h_partial(num_blocks);
  HIP_CHECK(hipMemcpy(h_partial.data(), d_dst, num_blocks * sizeof(TYPE),
                      hipMemcpyDeviceToHost));

  // Final reduction on CPU
  TYPE gpu_result = 0;
  for (int i = 0; i < num_blocks; i++) {
    gpu_result += h_partial[i];
  }

  // Compute reference on CPU
  TYPE cpu_result = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    cpu_result += h_src0[i] * h_src1[i];
  }

  // Verify result
  std::cout << "verify result" << std::endl;
  std::cout << "CPU result: " << cpu_result << std::endl;
  std::cout << "GPU result: " << gpu_result << std::endl;

  TYPE diff = std::abs(cpu_result - gpu_result);
  TYPE epsilon = FLOAT_ULP * std::numeric_limits<TYPE>::epsilon();

  bool passed = (diff < epsilon) || (diff / std::abs(cpu_result) < epsilon);

  // Cleanup
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_src1));
  HIP_CHECK(hipFree(d_dst));

  if (!passed) {
    std::cout << "Error: difference = " << diff << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
