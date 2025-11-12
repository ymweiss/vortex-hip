#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                 \
  do {                                                                 \
    hipError_t error = cmd;                                            \
    if (error != hipSuccess) {                                         \
      printf("error: '%s' returned %s!\n", #cmd,                       \
             hipGetErrorString(error));                                \
      exit(-1);                                                        \
    }                                                                  \
  } while (false)

#ifndef TYPE
#define TYPE float
#endif

// Random number generation utilities
unsigned int WangHash(unsigned int s) {
  s = (s^61) ^ (s >> 16);
  s *= 9;
  s = s ^ (s >> 4);
  s *= 0x27d4eb2d;
  s = s ^ (s >> 15);
  return s;
}

unsigned int RandomInt(unsigned int s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

float RandomFloat(unsigned int s) {
  return RandomInt(s) * 2.3283064365387e-10f;
}

// Kernel argument structure
typedef struct {
  uint32_t num_points;
  float dropout_p;
  float multiplier;
} kernel_arg_t;

// HIP Kernel
__global__ void dropout_kernel(const TYPE* src0, TYPE* dst,
                               uint32_t num_points,
                               float dropout_p, float multiplier) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_points) {
    float rand_value = RandomFloat(WangHash(idx));
    TYPE scaled_value = src0[idx] * multiplier;
    dst[idx] = (rand_value < dropout_p) ? 0.0 : scaled_value;
  }
}

// Comparator template for type-specific comparison
template <typename T>
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
    const int FLOAT_ULP = 6;
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

int main(int argc, char *argv[]) {
  uint32_t size = 16;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-n" && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      ++i;
    } else if (std::string(argv[i]) == "-h") {
      std::cout << "HIP Dropout Test" << std::endl;
      std::cout << "Usage: [-n size] [-h help]" << std::endl;
      return 0;
    }
  }

  std::srand(50);

  uint32_t num_points = size;
  float dropout_p = 0.2f;
  float multiplier = 1.0f / (1.0f - dropout_p);
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "dropout probability: " << dropout_p << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host memory
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Initialize source data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
  }

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE* d_src0 = nullptr;
  TYPE* d_dst = nullptr;

  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  std::cout << "dev_src0=0x" << std::hex << (uintptr_t)d_src0 << std::endl;
  std::cout << "dev_dst=0x" << std::hex << (uintptr_t)d_dst << std::dec << std::endl;

  // Copy data to device
  std::cout << "copy source buffer to device" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch dropout kernel" << std::endl;
  int blockSize = 256;
  int gridSize = (num_points + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(dropout_kernel, gridSize, blockSize, 0, 0,
                     d_src0, d_dst, num_points, dropout_p, multiplier);
  HIP_CHECK(hipGetLastError());

  // Wait for kernel completion
  std::cout << "wait for kernel completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back to host
  std::cout << "copy result buffer from device" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify results
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    float rand_value = RandomFloat(WangHash(i));

    auto ref = (rand_value < dropout_p) ? 0 : multiplier * h_src0[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // Clean up
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
