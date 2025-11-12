#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

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

typedef float TYPE;

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

// HIP kernel - dot product with reduction
__global__ void dotproduct_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points) {
  // Shared memory for partial sums
  extern __shared__ TYPE cache[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < num_points) {
    temp += src0[tid] * src1[tid];
    tid += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[cacheIndex] = temp;

  __syncthreads();

  // reduction in shared memory
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0)
    dst[blockIdx.x] = cache[0];
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 16;

  // Parse command line arguments
  int c;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-h") == 0) {
      std::cout << "HIP Dotproduct Test." << std::endl;
      std::cout << "Usage: [-n words] [-h: help]" << std::endl;
      exit(0);
    }
  }

  std::srand(50);

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  const uint32_t threadsPerBlock = 8;
  const uint32_t blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

  uint32_t dst_buf_size = blocksPerGrid * sizeof(TYPE);

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_src0, *d_src1, *d_dst;
  HIP_CHECK(hipMalloc(&d_src0, buf_size));
  HIP_CHECK(hipMalloc(&d_src1, buf_size));
  HIP_CHECK(hipMalloc(&d_dst, dst_buf_size));

  // Allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(blocksPerGrid);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
    h_src1[i] = Comparator<TYPE>::generate();
  }

  // Upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), buf_size, hipMemcpyHostToDevice));

  // Upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), buf_size, hipMemcpyHostToDevice));

  // Start device (launch kernel)
  std::cout << "start execution" << std::endl;
  hipLaunchKernelGGL(dotproduct_kernel,
                     dim3(blocksPerGrid),
                     dim3(threadsPerBlock),
                     threadsPerBlock * sizeof(TYPE), // shared memory size
                     0,
                     d_src0, d_src1, d_dst, num_points);
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, dst_buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  TYPE ref = 0;
  TYPE cur = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    ref += h_src0[i] * h_src1[i];
  }
  for (uint32_t i = 0; i < blocksPerGrid; i++)
    cur += h_dst[i];

  if (!Comparator<TYPE>::compare(cur, ref, 0, errors)) {
    ++errors;
  }

  // Cleanup
  std::cout << "cleanup" << std::endl;
  HIP_CHECK(hipFree(d_src0));
  HIP_CHECK(hipFree(d_src1));
  HIP_CHECK(hipFree(d_dst));

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
