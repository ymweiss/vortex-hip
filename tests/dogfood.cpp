#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>

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

// Kernel argument structure
typedef struct {
  uint32_t testid;
  uint32_t num_tasks;
  uint32_t task_size;
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
} kernel_arg_t;

///////////////////////////////////////////////////////////////////////////////

// Integer and float arithmetic kernels
__global__ void kernel_iadd(uint32_t* src0, uint32_t* src1, uint32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = ((int32_t*)src0)[offset + i];
    int32_t b = ((int32_t*)src1)[offset + i];
    int32_t c = a + b;
    ((int32_t*)dst)[offset + i] = c;
  }
}

__global__ void kernel_imul(uint32_t* src0, uint32_t* src1, uint32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = ((int32_t*)src0)[offset + i];
    int32_t b = ((int32_t*)src1)[offset + i];
    int32_t c = a * b;
    ((int32_t*)dst)[offset + i] = c;
  }
}

__global__ void kernel_idiv(uint32_t* src0, uint32_t* src1, uint32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = ((int32_t*)src0)[offset + i];
    int32_t b = ((int32_t*)src1)[offset + i];
    int32_t c = a / b;
    ((int32_t*)dst)[offset + i] = c;
  }
}

__global__ void kernel_idiv_mul(uint32_t* src0, uint32_t* src1, uint32_t* dst,
                                uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = ((int32_t*)src0)[offset + i];
    int32_t b = ((int32_t*)src1)[offset + i];
    int32_t c = a / b;
    int32_t d = a * b;
    int32_t e = c + d;
    ((int32_t*)dst)[offset + i] = e;
  }
}

__global__ void kernel_fadd(float* src0, float* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a + b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fsub(float* src0, float* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a - b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fmul(float* src0, float* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a * b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fmadd(float* src0, float* src1, float* dst,
                             uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a * b + b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fmsub(float* src0, float* src1, float* dst,
                             uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a * b - b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fnmadd(float* src0, float* src1, float* dst,
                              uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = -a * b - b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fnmsub(float* src0, float* src1, float* dst,
                              uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = -a * b + b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fnmadd_madd(float* src0, float* src1, float* dst,
                                   uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = -a * b - b;
    float d = a * b + b;
    float e = c + d;
    dst[offset + i] = e;
  }
}

__global__ void kernel_fdiv(float* src0, float* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a / b;
    dst[offset + i] = c;
  }
}

__global__ void kernel_fdiv2(float* src0, float* src1, float* dst,
                             uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a / b;
    float d = b / a;
    float e = c + d;
    dst[offset + i] = e;
  }
}

__global__ void kernel_fsqrt(float* src0, float* src1, float* dst,
                             uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = sqrtf(a * b);
    dst[offset + i] = c;
  }
}

__global__ void kernel_ftoi(float* src0, float* src1, int32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a + b;
    int32_t d = (int32_t)c;
    dst[offset + i] = d;
  }
}

__global__ void kernel_ftou(float* src0, float* src1, uint32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    float c = a + b;
    uint32_t d = (uint32_t)c;
    dst[offset + i] = d;
  }
}

__global__ void kernel_itof(int32_t* src0, int32_t* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = src0[offset + i];
    int32_t b = src1[offset + i];
    int32_t c = a + b;
    float d = (float)c;
    dst[offset + i] = d;
  }
}

__global__ void kernel_utof(int32_t* src0, int32_t* src1, float* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int32_t a = src0[offset + i];
    int32_t b = src1[offset + i];
    int32_t c = a + b;
    float d = (float)c;
    dst[offset + i] = d;
  }
}

__device__ float fclamp(float a, float b, float c) {
  return fminf(fmaxf(a, b), c);
}

__global__ void kernel_fclamp(float* src0, float* src1, float* dst,
                              uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    float a = src0[offset + i];
    float b = src1[offset + i];
    dst[offset + i] = fclamp(1.0f, a, b);
  }
}

__device__ int iclamp(int a, int b, int c) {
  return min(max(a, b), c);
}

__global__ void kernel_iclamp(int* src0, int* src1, int* dst,
                              uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    int a = src0[offset + i];
    int b = src1[offset + i];
    dst[offset + i] = iclamp(1, a, b);
  }
}

__global__ void kernel_trigo(float* src0, float* src1, float* dst,
                             uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t offset = bid * count;

  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    uint32_t j = offset + i;
    float a = src0[j];
    float b = src1[j];
    float c = a * b;
    if ((j % 4) == 0) {
      c = sinf(c);
    }
    dst[j] = c;
  }
}

__global__ void kernel_bar(uint32_t* src0, uint32_t* dst,
                           uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;
  uint32_t block_size = num_tasks / gridDim.x;
  uint32_t offset = bid * block_size;

  // Update destination using first thread
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < block_size; ++i) {
      dst[i + offset] = src0[i + offset];
    }
  }

  // Synchronize within block
  __syncthreads();

  // Update destination
  dst[bid] += 1;
}

__global__ void kernel_gbar(uint32_t* src0, uint32_t* dst,
                            uint32_t count, uint32_t num_tasks) {
  uint32_t bid = blockIdx.x;

  // Update destination using first thread in entire grid
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (uint32_t i = 0; i < num_tasks; ++i) {
      dst[i] = src0[i];
    }
  }

  // Synchronize all blocks (global sync)
  __syncthreads();

  // Update destination
  dst[bid] += 1;
}

///////////////////////////////////////////////////////////////////////////////

// Test data generators
void generate_int_data(std::vector<int32_t>& src0, std::vector<int32_t>& src1) {
  for (size_t i = 0; i < src0.size(); ++i) {
    src0[i] = (int32_t)(rand() % 256);
    src1[i] = (int32_t)(rand() % 256 + 1); // avoid division by zero
  }
}

void generate_float_data(std::vector<float>& src0, std::vector<float>& src1) {
  for (size_t i = 0; i < src0.size(); ++i) {
    src0[i] = (float)(rand() % 100) / 10.0f;
    src1[i] = (float)(rand() % 100 + 1) / 10.0f; // avoid division by zero
  }
}

// Verification functions
bool verify_int_add(const std::vector<int32_t>& dst,
                    const std::vector<int32_t>& src0,
                    const std::vector<int32_t>& src1,
                    uint32_t num_tasks, uint32_t count) {
  for (uint32_t b = 0; b < num_tasks / count; ++b) {
    uint32_t offset = b * count;
    for (uint32_t i = 0; i < count; ++i) {
      int32_t expected = src0[offset + i] + src1[offset + i];
      if (dst[offset + i] != expected) {
        std::cout << "Mismatch at [" << offset + i << "]: expected="
                  << expected << ", got=" << dst[offset + i] << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verify_float_add(const std::vector<float>& dst,
                      const std::vector<float>& src0,
                      const std::vector<float>& src1,
                      uint32_t num_tasks, uint32_t count) {
  for (uint32_t b = 0; b < num_tasks / count; ++b) {
    uint32_t offset = b * count;
    for (uint32_t i = 0; i < count; ++i) {
      float expected = src0[offset + i] + src1[offset + i];
      float actual = dst[offset + i];
      if (std::abs(actual - expected) > 1e-5f) {
        std::cout << "Mismatch at [" << offset + i << "]: expected="
                  << expected << ", got=" << actual << std::endl;
        return false;
      }
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t count = 64;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      count = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  // Get device properties
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return 1;
  }

  std::cout << "number of HIP devices: " << device_count << std::endl;

  // Set device
  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t dev_props;
  HIP_CHECK(hipGetDeviceProperties(&dev_props, 0));
  std::cout << "using device: " << dev_props.name << std::endl;

  uint32_t num_tasks = 256; // Number of work items
  uint32_t num_points = count * num_tasks;
  size_t buf_size = num_points * sizeof(uint32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Test 1: Integer addition
  std::cout << "\n=== Test 1: Integer Addition ===" << std::endl;

  std::vector<int32_t> h_int_src0(num_points);
  std::vector<int32_t> h_int_src1(num_points);
  std::vector<int32_t> h_int_dst(num_points, 0);

  generate_int_data(h_int_src0, h_int_src1);

  int32_t *d_int_src0, *d_int_src1, *d_int_dst;
  HIP_CHECK(hipMalloc(&d_int_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_int_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_int_dst, buf_size));

  HIP_CHECK(hipMemcpy(d_int_src0, h_int_src0.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_int_src1, h_int_src1.data(), buf_size, hipMemcpyHostToDevice));

  int block_size = 256;
  int num_blocks = (num_tasks + block_size - 1) / block_size;
  hipLaunchKernelGGL(kernel_iadd, dim3(num_blocks), dim3(block_size), 0, 0,
                     (uint32_t*)d_int_src0, (uint32_t*)d_int_src1, (uint32_t*)d_int_dst,
                     count, num_tasks);

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_int_dst.data(), d_int_dst, buf_size, hipMemcpyDeviceToHost));

  if (verify_int_add(h_int_dst, h_int_src0, h_int_src1, num_tasks, count)) {
    std::cout << "Test 1 PASSED!" << std::endl;
  } else {
    std::cout << "Test 1 FAILED!" << std::endl;
  }

  HIP_CHECK(hipFree(d_int_src0));
  HIP_CHECK(hipFree(d_int_src1));
  HIP_CHECK(hipFree(d_int_dst));

  // Test 2: Float addition
  std::cout << "\n=== Test 2: Float Addition ===" << std::endl;

  std::vector<float> h_float_src0(num_points);
  std::vector<float> h_float_src1(num_points);
  std::vector<float> h_float_dst(num_points, 0.0f);

  generate_float_data(h_float_src0, h_float_src1);

  float *d_float_src0, *d_float_src1, *d_float_dst;
  HIP_CHECK(hipMalloc(&d_float_src0, num_points * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_float_src1, num_points * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_float_dst, num_points * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_float_src0, h_float_src0.data(), num_points * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_float_src1, h_float_src1.data(), num_points * sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(kernel_fadd, dim3(num_blocks), dim3(block_size), 0, 0,
                     d_float_src0, d_float_src1, d_float_dst, count, num_tasks);

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_float_dst.data(), d_float_dst, num_points * sizeof(float),
                      hipMemcpyDeviceToHost));

  if (verify_float_add(h_float_dst, h_float_src0, h_float_src1, num_tasks, count)) {
    std::cout << "Test 2 PASSED!" << std::endl;
  } else {
    std::cout << "Test 2 FAILED!" << std::endl;
  }

  HIP_CHECK(hipFree(d_float_src0));
  HIP_CHECK(hipFree(d_float_src1));
  HIP_CHECK(hipFree(d_float_dst));

  std::cout << "\n=== All Tests Complete ===" << std::endl;

  return 0;
}
