#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) do { hipError_t status = cmd; if (status != hipSuccess) { printf("Error: HIP call failed with code %d: %s\n", status, hipGetErrorString(status)); cleanup(); exit(-1); } } while(false)

#define FLOAT_ULP 6

// Type definition - can be overridden at compile time
#ifndef TYPE
#define TYPE int
#endif

// Kernel argument structure
typedef struct {
  uint32_t num_tasks;
  uint32_t task_size;
  TYPE* src0_ptr;
  TYPE* src1_ptr;
  TYPE* dst_ptr;
} kernel_arg_t;

///////////////////////////////////////////////////////////////////////////////
// HIP Kernel
///////////////////////////////////////////////////////////////////////////////

__global__ void kernel_add(const TYPE* src0, const TYPE* src1, TYPE* dst,
                           uint32_t count, uint32_t num_blocks) {
  uint32_t block_idx = blockIdx.x;
  uint32_t thread_idx = threadIdx.x;
  uint32_t total_threads = blockDim.x * gridDim.x;

  // Each block processes 'count' elements
  uint32_t offset = block_idx * count;

  // Distribute iterations across all threads in the block
  for (uint32_t i = thread_idx; i < count; i += blockDim.x) {
    dst[offset + i] = src0[offset + i] + src1[offset + i];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Host Code
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
        printf("*** error: [%d] expected=%f(0x%x), actual=%f(0x%x), ulp=%d\n", index, b, fb.i, a, fa.i, d);
      }
      return false;
    }
    return true;
  }
};

// Global state
uint32_t count = 16;
TYPE* d_src0 = nullptr;
TYPE* d_src1 = nullptr;
TYPE* d_dst = nullptr;

static void show_usage() {
   std::cout << "HIP Demo Test." << std::endl;
   std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'h':{
      show_usage();
      exit(0);
    } break;
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
  if (d_src1) {
    HIP_CHECK(hipFree(d_src1));
    d_src1 = nullptr;
  }
  if (d_dst) {
    HIP_CHECK(hipFree(d_dst));
    d_dst = nullptr;
  }
}

int main(int argc, char *argv[]) {
  // Parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // Get device info
  std::cout << "Getting HIP device info" << std::endl;
  int device_id = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
  HIP_CHECK(hipSetDevice(device_id));

  // Calculate grid configuration
  uint32_t num_blocks = props.multiProcessorCount;
  uint32_t num_threads_per_block = 256;
  uint32_t total_threads = num_blocks;
  uint32_t num_points = count * total_threads;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "number of blocks: " << num_blocks << std::endl;
  std::cout << "threads per block: " << num_threads_per_block << std::endl;
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points);

  // Generate source data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
    h_src1[i] = Comparator<TYPE>::generate();
  }

  // Upload source buffers
  std::cout << "upload source buffer0" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  std::cout << "upload source buffer1" << std::endl;
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "launch kernel" << std::endl;
  hipLaunchKernelGGL(kernel_add,
                     dim3(num_blocks),
                     dim3(num_threads_per_block),
                     0, 0,
                     d_src0, d_src1, d_dst, count, num_blocks);
  HIP_CHECK(hipGetLastError());

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto ref = h_src0[i] + h_src1[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
