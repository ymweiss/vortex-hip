#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>

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

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < width; ++e) {
          sum += A[row * width + e] * B[e * width + col];
      }
      out[row * width + col] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for matrix multiplication
__global__ void sgemm_kernel(TYPE* A, TYPE* B, TYPE* C, uint32_t size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < size && row < size) {
    TYPE sum(0);
    for (int e = 0; e < size; ++e) {
      sum += A[row * size + e] * B[e * size + col];
    }
    C[row * size + col] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 32;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t size_sq = size * size;
  uint32_t buf_size = size_sq * sizeof(TYPE);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;

  // Generate source data
  std::vector<TYPE> h_A(size_sq);
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C(size_sq);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = Comparator<TYPE>::generate();
    h_B[i] = Comparator<TYPE>::generate();
  }

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  TYPE *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, buf_size));
  HIP_CHECK(hipMalloc(&d_B, buf_size));
  HIP_CHECK(hipMalloc(&d_C, buf_size));

  // Upload matrix A buffer
  std::cout << "upload matrix A buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));

  // Upload matrix B buffer
  std::cout << "upload matrix B buffer" << std::endl;
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  auto time_start = std::chrono::high_resolution_clock::now();

  // Launch kernel
  std::cout << "start device" << std::endl;
  dim3 blockSize(16, 16);
  dim3 numBlocks((size + blockSize.x - 1) / blockSize.x,
                 (size + blockSize.y - 1) / blockSize.y);
  hipLaunchKernelGGL(sgemm_kernel, numBlocks, blockSize, 0, 0,
                     d_A, d_B, d_C, size);

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  HIP_CHECK(hipDeviceSynchronize());

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Download destination buffer
  std::cout << "download destination buffer" << std::endl;
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
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
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
