#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

#define NUM_LOADS 8

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

typedef struct {
  uint32_t num_tasks;
  uint32_t size;
  uint32_t stride;
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
} kernel_arg_t;

///////////////////////////////////////////////////////////////////////////////

union Float_t {
  float f;
  int   i;
  struct {
    uint32_t man  : 23;
    uint32_t exp  : 8;
    uint32_t sign : 1;
  } parts;
};

inline float fround(float x, int32_t precision = 8) {
  auto power_of_10 = std::pow(10, precision);
  return std::round(x * power_of_10) / power_of_10;
}

inline bool almost_equal_eps(float a, float b, int ulp = 128) {
  auto eps = std::numeric_limits<float>::epsilon() * (std::max(fabs(a), fabs(b)) * ulp);
  auto d = fabs(a - b);
  if (d > eps) {
    std::cout << "*** almost_equal_eps: d=" << d << ", eps=" << eps << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal_ulp(float a, float b, int32_t ulp = 6) {
  Float_t fa{a}, fb{b};
  auto d = std::abs(fa.i - fb.i);
  if (d > ulp) {
    std::cout << "*** almost_equal_ulp: a=" << a << ", b=" << b << ", ulp=" << d << ", ia=" << std::hex << fa.i << ", ib=" << fb.i << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal(float a, float b) {
  if (a == b)
    return true;
  return almost_equal_ulp(a, b);
}

///////////////////////////////////////////////////////////////////////////////

// HIP kernel - memory stress test
__global__ void mstress_kernel(uint32_t stride, uint32_t* addr_ptr, float* src_ptr, float* dst_ptr) {
  uint32_t offset = blockIdx.x * stride;

  for (uint32_t i = 0; i < stride; ++i) {
    float value = 0.0f;
    for (uint32_t j = 0; j < NUM_LOADS; ++j) {
      uint32_t addr = offset + i + j;
      uint32_t index = addr_ptr[addr];
      value *= src_ptr[index];
    }
    dst_ptr[offset + i] = value;
  }
}

///////////////////////////////////////////////////////////////////////////////

static void show_usage() {
  std::cout << "HIP Memory Stress Test." << std::endl;
  std::cout << "Usage: [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t& count) {
  int c;
  int argc_idx = 1;
  while (argc_idx < argc) {
    if (strcmp(argv[argc_idx], "-n") == 0 && argc_idx + 1 < argc) {
      count = atoi(argv[argc_idx + 1]);
      argc_idx += 2;
    } else if (strcmp(argv[argc_idx], "-h") == 0) {
      show_usage();
      exit(0);
    } else {
      show_usage();
      exit(-1);
    }
  }
}

void gen_src_data(std::vector<float>& test_data,
                  std::vector<uint32_t>& addr_table,
                  uint32_t num_points,
                  uint32_t num_addrs) {
  test_data.resize(num_points);
  addr_table.resize(num_addrs);

  for (uint32_t i = 0; i < num_points; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    test_data[i] = r;
  }

  for (uint32_t i = 0; i < num_addrs; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    uint32_t index = static_cast<uint32_t>(r * num_points);
    if (index >= num_points) index = num_points - 1;
    addr_table[i] = index;
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t count = 0;

  // parse command arguments
  parse_args(argc, argv, count);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // Get device properties
  std::cout << "Getting device properties" << std::endl;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  // Use multiProcessorCount as equivalent to num_cores
  uint32_t num_cores = prop.multiProcessorCount;
  uint32_t num_warps = 1;  // Simplified for HIP
  uint32_t num_threads = prop.warpSize * prop.maxThreadsPerBlock / prop.warpSize;

  uint32_t total_threads = num_cores * num_warps * num_threads;
  uint32_t num_points = count * total_threads;
  uint32_t num_addrs = num_points + NUM_LOADS - 1;

  uint32_t addr_buf_size = num_addrs * sizeof(uint32_t);
  uint32_t src_buf_size = num_points * sizeof(float);
  uint32_t dst_buf_size = num_points * sizeof(float);

  std::cout << "number of cores: " << num_cores << std::endl;
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;
  std::cout << "src buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "dst buffer size: " << dst_buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  uint32_t* d_addr = nullptr;
  float* d_src = nullptr;
  float* d_dst = nullptr;

  HIP_CHECK(hipMalloc(&d_addr, addr_buf_size));
  HIP_CHECK(hipMalloc(&d_src, src_buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<uint32_t> h_addr;
  std::vector<float> h_src;
  std::vector<float> h_dst(num_points);
  gen_src_data(h_src, h_addr, num_points, num_addrs);

  // upload address buffer
  std::cout << "upload address buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_addr, h_addr.data(), addr_buf_size, hipMemcpyHostToDevice));

  // upload source buffer
  std::cout << "upload source buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_buf_size, hipMemcpyHostToDevice));

  // start device
  std::cout << "start kernel execution" << std::endl;
  hipLaunchKernelGGL(mstress_kernel,
                     dim3(total_threads),  // number of blocks
                     dim3(1),              // threads per block
                     0,                    // shared memory
                     0,                    // stream
                     count, d_addr, d_src, d_dst);

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    float ref = 0.0f;
    for (uint32_t j = 0; j < NUM_LOADS; ++j) {
      uint32_t addr = i + j;
      uint32_t index = h_addr[addr];
      float value = h_src[index];
      ref *= value;
    }

    float cur = h_dst[i];
    if (!almost_equal(cur, ref)) {
      std::cout << "error at result #" << std::dec << i
                << ": actual " << cur << ", expected " << ref << std::endl;
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_addr));
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
