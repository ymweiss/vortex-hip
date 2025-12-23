// SGEMV (Matrix-Vector Multiplication) - HIP version
// Converted from Vortex regression test: vortex/tests/regression/sgemv

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
    if (fabs(a - b) > 1e-3f) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for matrix-vector multiplication
// Computes y = A * x where A is M x N matrix, x is N x 1 vector, y is M x 1 vector
__global__ void sgemv_kernel(TYPE* A, TYPE* x, TYPE* y, uint32_t M, uint32_t N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M) {
    TYPE sum = 0.0f;
    for (int col = 0; col < N; ++col) {
      sum += A[row * N + col] * x[col];
    }
    y[row] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

// CPU reference implementation
void sgemv_cpu(TYPE* y, const TYPE* A, const TYPE* x, uint32_t M, uint32_t N) {
  for (uint32_t i = 0; i < M; ++i) {
    TYPE sum = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t M = 256;  // Rows (output vector size)
  uint32_t N = 256;  // Columns (input vector size)

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      M = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      N = atoi(argv[i + 1]);
      i++;
    }
  }

  std::srand(50);

  uint32_t A_size = M * N * sizeof(TYPE);
  uint32_t x_size = N * sizeof(TYPE);
  uint32_t y_size = M * sizeof(TYPE);

  std::cout << "Matrix dimensions: " << M << " x " << N << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "Matrix size: " << A_size << " bytes" << std::endl;
  std::cout << "Vector x size: " << x_size << " bytes" << std::endl;
  std::cout << "Vector y size: " << y_size << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<TYPE> h_A(M * N);
  std::vector<TYPE> h_x(N);
  std::vector<TYPE> h_y(M);

  // Generate synthetic data
  for (uint32_t i = 0; i < M * N; ++i) {
    h_A[i] = Comparator<TYPE>::generate();
  }
  for (uint32_t i = 0; i < N; ++i) {
    h_x[i] = Comparator<TYPE>::generate();
  }

  // Allocate device memory
  printf("allocate device memory\n");
  TYPE *d_A, *d_x, *d_y;
  HIP_CHECK(hipMalloc(&d_A, A_size));
  HIP_CHECK(hipMalloc(&d_x, x_size));
  HIP_CHECK(hipMalloc(&d_y, y_size));

  // Upload input buffers
  printf("upload matrix A\n");
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));

  printf("upload vector x\n");
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  // Note: Vortex supports max 16 threads per block
  int threads_per_block = 16;
  int num_blocks = (M + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(sgemv_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                     d_A, d_x, d_y, M, N);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  std::vector<TYPE> h_ref(M);
  sgemv_cpu(h_ref.data(), h_A.data(), h_x.data(), M, N);

  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    auto ref = h_ref[i];
    auto cur = h_y[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
