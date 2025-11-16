// Memory Fence Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/fence
// Tests memory fence operations to ensure proper memory ordering

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#ifndef TYPE
#define TYPE int32_t
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

// HIP kernel with memory fence
__global__ void fence_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t task_size, uint32_t num_tasks) {
  uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (task_id >= num_tasks) return;

  uint32_t offset = task_id * task_size;
  for (uint32_t i = 0; i < task_size; ++i) {
    dst[offset + i] = src0[offset + i] + src1[offset + i];
  }

  // Memory fence - ensures all writes are visible
  __threadfence();
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t num_tasks = 4;
  uint32_t task_size = 16;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      num_tasks = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      task_size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t num_points = num_tasks * task_size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of tasks: " << num_tasks << std::endl;
  std::cout << "task size: " << task_size << std::endl;
  std::cout << "total points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Initialize with random values
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = rand() % 100;
    h_src1[i] = rand() % 100;
  }

  // Allocate device memory
  TYPE *d_src0, *d_src1, *d_dst;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Upload source buffers
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  int threads_per_block = 64;
  int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(fence_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src0, d_src1, d_dst, task_size, num_tasks);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    TYPE ref = h_src0[i] + h_src1[i];
    TYPE cur = h_dst[i];
    if (cur != ref) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      }
      ++errors;
    }
  }

  // Cleanup
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
