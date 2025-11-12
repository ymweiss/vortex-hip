#include <hip/hip_runtime.h>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(expr) \
  do { \
    hipError_t _ret = expr; \
    if (_ret != hipSuccess) { \
      printf("Error: '%s' returned %s!\n", #expr, hipGetErrorString(_ret)); \
      cleanup(); \
      exit(-1); \
    } \
  } while (false)

#define FLOAT_ULP 6

#ifndef TYPE
#define TYPE float
#endif

///////////////////////////////////////////////////////////////////////////////
// Data structures
///////////////////////////////////////////////////////////////////////////////

typedef struct {
  uint32_t num_points;
  uint64_t src0_addr;
  uint64_t dst_addr;
} kernel_arg_t;

///////////////////////////////////////////////////////////////////////////////
// Global variables
///////////////////////////////////////////////////////////////////////////////

uint32_t size = 16;
TYPE* d_src0 = nullptr;
TYPE* d_dst = nullptr;
kernel_arg_t kernel_arg = {};

///////////////////////////////////////////////////////////////////////////////
// HIP Kernel
///////////////////////////////////////////////////////////////////////////////

__global__ void relu_kernel(TYPE* src0_ptr, TYPE* dst_ptr, uint32_t num_points) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_points) {
    TYPE value = src0_ptr[idx];
    dst_ptr[idx] = (value < 0) ? 0 : value;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Template Comparators
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
private:
  union Float_t { float f; int i; };
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
// Utility functions
///////////////////////////////////////////////////////////////////////////////

static void show_usage() {
  std::cout << "HIP ReLU Test." << std::endl;
  std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (d_src0) {
    HIP_CHECK(hipFree(d_src0));
    d_src0 = nullptr;
  }
  if (d_dst) {
    HIP_CHECK(hipFree(d_dst));
    d_dst = nullptr;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // check HIP device
  std::cout << "checking HIP device" << std::endl;
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP devices found!" << std::endl;
    return 1;
  }

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.num_points = num_points;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  std::cout << "dev_src0=0x" << std::hex << (uint64_t)d_src0 << std::endl;
  std::cout << "dev_dst=0x" << std::hex << (uint64_t)d_dst << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_dst(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
  }

  // upload source buffer
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // launch kernel
  std::cout << "launch kernel" << std::endl;
  int blockSize = 256;
  int numBlocks = (num_points + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(relu_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, d_src0, d_dst, num_points);
  HIP_CHECK(hipGetLastError());

  // wait for kernel completion
  std::cout << "wait for kernel completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto ref = (h_src0[i] < 0) ? 0 : h_src0[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
