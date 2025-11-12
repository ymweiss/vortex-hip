// HIP host code for tiled matrix multiplication (SGEMM2)
// Adapted from tests/sgemm2.cpp for Vortex HIP runtime

#include "vortex_hip_runtime.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while (0)

#define FLOAT_ULP 6

extern void* kernel_body_handle;

// CPU reference implementation
static void matmul_cpu(float* out, const float* A, const float* B, uint32_t size) {
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
      float sum = 0.0f;
      for (uint32_t e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
      }
      out[row * size + col] = sum;
    }
  }
}

// Float comparison with ULP tolerance
static bool compare_float(float a, float b, int index, int& errors) {
  union { float f; int32_t i; } fa, fb;
  fa.f = a;
  fb.f = b;
  auto d = std::abs(fa.i - fb.i);
  if (d > FLOAT_ULP) {
    if (errors < 10) {
      std::cout << "*** error: [" << index << "] expected=" << b
                << ", actual=" << a << ", ulp_diff=" << d << std::endl;
    }
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  uint32_t size = 16;
  uint32_t tile_size = 4;

  // Parse command line
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      tile_size = atoi(argv[i + 1]);
      i++;
    }
  }

  // Validate that size is a multiple of tile_size
  if ((size / tile_size) * tile_size != size) {
    std::cerr << "Error: matrix size " << size
              << " must be a multiple of tile size " << tile_size << std::endl;
    return 1;
  }

  std::srand(50);

  uint32_t size_sq = size * size;
  uint32_t buf_size = size_sq * sizeof(float);
  uint32_t num_blocks = size / tile_size;
  uint32_t shared_mem_bytes = 2 * tile_size * tile_size * sizeof(float);

  std::cout << "data type: float" << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "tile size: " << tile_size << "x" << tile_size << std::endl;
  std::cout << "grid: " << num_blocks << "x" << num_blocks << " blocks" << std::endl;
  std::cout << "block: " << tile_size << "x" << tile_size << " threads" << std::endl;
  std::cout << "shared memory: " << shared_mem_bytes << " bytes" << std::endl;

  // Initialize device
  std::cout << "initialize HIP device" << std::endl;
  HIP_CHECK(hipSetDevice(0));

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  float *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc((void**)&d_A, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_B, buf_size));
  HIP_CHECK(hipMalloc((void**)&d_C, buf_size));

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<float> h_A(size_sq);
  std::vector<float> h_B(size_sq);
  std::vector<float> h_C(size_sq);

  // Generate source data
  std::cout << "generate source data" << std::endl;
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Upload source buffers
  std::cout << "upload source buffers" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  // Only pass 4 arguments to match metadata: A, B, C, size
  // tile_size is derived from block_dim in the kernel
  void* args[] = {&d_A, &d_B, &d_C, &size};

  HIP_CHECK(hipLaunchKernel(kernel_body_handle,
                            dim3(num_blocks, num_blocks),
                            dim3(tile_size, tile_size),
                            args,
                            shared_mem_bytes,
                            nullptr));

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<float> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size);

    for (uint32_t i = 0; i < size_sq; ++i) {
      if (!compare_float(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
