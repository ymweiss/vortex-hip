#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

// HIP error checking macro
#define HIP_CHECK(cmd)                                                \
  do {                                                                \
    hipError_t error = cmd;                                           \
    if (error != hipSuccess) {                                        \
      printf("HIP Error: %s\n", hipGetErrorString(error));            \
      exit(-1);                                                       \
    }                                                                 \
  } while (false)

// Kernel argument structure
typedef struct {
  uint32_t num_tasks;
  uint32_t task_size;
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
} kernel_arg_t;

// HIP kernel - performs element-wise addition with fence
__global__ void fence_kernel(int32_t* src0, int32_t* src1, int32_t* dst,
                             uint32_t count) {
  uint32_t offset = blockIdx.x * count;
  for (uint32_t i = 0; i < count; ++i) {
    dst[offset + i] = src0[offset + i] + src1[offset + i];
  }
  __threadfence();  // Ensure all writes are visible to other threads
}

// Global variables for cleanup
int32_t* d_src0 = nullptr;
int32_t* d_src1 = nullptr;
int32_t* d_dst = nullptr;

void cleanup() {
  if (d_src0) HIP_CHECK(hipFree(d_src0));
  if (d_src1) HIP_CHECK(hipFree(d_src1));
  if (d_dst) HIP_CHECK(hipFree(d_dst));
}

static void show_usage() {
  std::cout << "HIP Fence Test." << std::endl;
  std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char** argv, uint32_t& count) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
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

int main(int argc, char* argv[]) {
  uint32_t count = 0;

  // parse command arguments
  parse_args(argc, argv, count);

  if (count == 0) {
    count = 1;
  }

  // Get device properties
  std::cout << "Getting device properties" << std::endl;
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));

  uint32_t num_blocks = deviceProp.multiProcessorCount;
  uint32_t num_threads_per_block = deviceProp.maxThreadsPerBlock;
  uint32_t total_threads = num_blocks;

  uint32_t num_points = count * total_threads;
  size_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of blocks: " << num_blocks << std::endl;
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, buf_size));

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int32_t> h_src0(num_points);
  std::vector<int32_t> h_src1(num_points);
  std::vector<int32_t> h_dst(num_points);

  // generate source data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = i - 1;
    h_src1[i] = i + 1;
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // launch kernel
  std::cout << "launch kernel" << std::endl;
  hipLaunchKernelGGL(fence_kernel,
                     dim3(num_blocks),
                     dim3(1),
                     0,
                     0,
                     d_src0, d_src1, d_dst, count);
  HIP_CHECK(hipGetLastError());

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    int32_t ref = i + i;
    int32_t cur = h_dst[i];
    if (cur != ref) {
      std::cout << "error at result #" << std::dec << i
                << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
      ++errors;
    }
  }

  // cleanup
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
