// Dropout Neural Network Layer - HIP version
// Converted from Vortex regression test: vortex/tests/regression/dropout

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>

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

// Random number generation functions (from common.h)
// Use __host__ __device__ so they can be called from both device kernel and host verification
__host__ __device__ unsigned int WangHash(unsigned int s) {
  s = (s ^ 61) ^ (s >> 16);
  s *= 9;
  s = s ^ (s >> 4);
  s *= 0x27d4eb2d;
  s = s ^ (s >> 15);
  return s;
}

__host__ __device__ unsigned int RandomInt(unsigned int s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

__host__ __device__ float RandomFloat(unsigned int s) {
  return RandomInt(s) * 2.3283064365387e-10f;
}

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "integer";
  }
  static int generate() {
    return rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

// HIP kernel
__global__ void dropout_kernel(TYPE* src0, TYPE* dst, uint32_t num_points,
                               float dropout_p, float multiplier) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points) {
    float rand_value = RandomFloat(WangHash(idx));
    TYPE scaled_value = src0[idx] * multiplier;
    dst[idx] = (rand_value < dropout_p) ? 0.0 : scaled_value;
  }
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
  float dropout_p = 0.2f;
  float multiplier = 1.0f / (1.0f - dropout_p);
  uint32_t buf_size = num_points * sizeof(TYPE);

  printf("number of points: %u\n", num_points);
  printf("dropout probability: %u\n", dropout_p);
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_dst(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
  }

  // Allocate device memory
  printf("allocate device memory\n");
  TYPE *d_src0, *d_dst;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Upload source buffer
  printf("upload source buffer0\n");
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  int threads_per_block = 64;
  int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(dropout_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_src0, d_dst, num_points, dropout_p, multiplier);

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
    float rand_value = RandomFloat(WangHash(i));
    auto ref = (rand_value < dropout_p) ? 0 : multiplier * h_src0[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
