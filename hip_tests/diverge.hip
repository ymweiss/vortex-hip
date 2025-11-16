// Divergence Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/diverge
// Tests control flow divergence patterns: branches, loops, switches

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

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

struct key_t {
  uint32_t user = 0;
};

__device__ __noinline__ void hacker(key_t* key, uint32_t task_id) {
  key->user = task_id;
}

// HIP kernel testing various divergence patterns
__global__ void diverge_kernel(int32_t* src, int32_t* dst, uint32_t num_points, uint32_t samples) {
  uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (task_id >= num_points) return;

  int value = src[task_id];

  key_t key;
  uint32_t sample_count = samples;
  while (sample_count--) {
    hacker(&key, task_id);
    if ((key.user & 0x1) == 0) {
      value += 1;
    }
  }

  // None taken branch
  if (task_id >= 0x7fffffff) {
    value = 0;
  } else {
    value += 2;
  }

  // Divergent branches
  if (task_id > 1) {
    if (task_id > 2) {
      value += 6;
    } else {
      value += 5;
    }
  } else {
    if (task_id > 0) {
      value += 4;
    } else {
      value += 3;
    }
  }

  // All taken branch
  if (task_id >= 0) {
    value += 7;
  } else {
    value = 0;
  }

  // Divergent loop
  for (int i = 0, n = task_id; i < n; ++i) {
    value += src[i];
  }

  // Switch statement
  switch (task_id) {
  case 0:
    value += 1;
    break;
  case 1:
    value -= 1;
    break;
  case 2:
    value *= 3;
    break;
  case 3:
    value *= 5;
    break;
  default:
    break;
  }

  // Ternary select
  value += (task_id >= 0) ? ((task_id > 5) ? src[0] : task_id) : ((task_id < 5) ? src[1] : -task_id);

  // Min/max
  value += min(src[task_id], value);
  value += max(src[task_id], value);

  dst[task_id] = value;
}

///////////////////////////////////////////////////////////////////////////////

// CPU reference implementation
int32_t compute_reference(int32_t* src, uint32_t task_id, uint32_t samples) {
  int value = src[task_id];

  key_t key;
  uint32_t sample_count = samples;
  while (sample_count--) {
    key.user = task_id;  // Inline hacker function
    if ((key.user & 0x1) == 0) {
      value += 1;
    }
  }

  if (task_id >= 0x7fffffff) {
    value = 0;
  } else {
    value += 2;
  }

  if (task_id > 1) {
    if (task_id > 2) {
      value += 6;
    } else {
      value += 5;
    }
  } else {
    if (task_id > 0) {
      value += 4;
    } else {
      value += 3;
    }
  }

  if (task_id >= 0) {
    value += 7;
  } else {
    value = 0;
  }

  for (int i = 0, n = task_id; i < n; ++i) {
    value += src[i];
  }

  switch (task_id) {
  case 0:
    value += 1;
    break;
  case 1:
    value -= 1;
    break;
  case 2:
    value *= 3;
    break;
  case 3:
    value *= 5;
    break;
  default:
    break;
  }

  value += (task_id >= 0) ? ((task_id > 5) ? src[0] : task_id) : ((task_id < 5) ? src[1] : -task_id);

  value += std::min(src[task_id], value);
  value += std::max(src[task_id], value);

  return value;
}

int main(int argc, char *argv[]) {
  uint32_t size = 16;
  uint32_t samples = 10;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "samples: " << samples << std::endl;

  // Allocate host buffers
  std::vector<int32_t> h_src(num_points);
  std::vector<int32_t> h_dst(num_points);

  // Initialize with random values
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = rand() % 100;
  }

  // Allocate device memory
  int32_t *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Upload source buffer
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(diverge_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src, d_dst, num_points, samples);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    int32_t ref = compute_reference(h_src.data(), i, samples);
    int32_t cur = h_dst[i];
    if (cur != ref) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      }
      ++errors;
    }
  }

  // Cleanup
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
