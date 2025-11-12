#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// HIP error checking macro
#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t error = (call);                                               \
    if (hipSuccess != error) {                                               \
      printf("HIP error '%s' at line %d: %s\n", #call, __LINE__,             \
             hipGetErrorString(error));                                       \
      exit(-1);                                                              \
    }                                                                         \
  } while (false)

// Kernel argument structure
typedef struct {
  uint32_t M;
  uint32_t N;
} kernel_arg_t;

// HIP kernel for SGEMV
__global__ void kernel_sgemv(float *A, float *x, float *y, uint32_t M, uint32_t N) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  float sum = 0.0f;
  for (uint32_t col = 0; col < N; col += 4) {
    // Load 4 elements from A and x
    float a0 = A[row * N + col];
    float a1 = (col + 1 < N) ? A[row * N + col + 1] : 0.0f;
    float a2 = (col + 2 < N) ? A[row * N + col + 2] : 0.0f;
    float a3 = (col + 3 < N) ? A[row * N + col + 3] : 0.0f;

    float b0 = x[col];
    float b1 = (col + 1 < N) ? x[col + 1] : 0.0f;
    float b2 = (col + 2 < N) ? x[col + 2] : 0.0f;
    float b3 = (col + 3 < N) ? x[col + 3] : 0.0f;

    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  }
  y[row] = sum;
}

// CPU reference implementation
void sgemv_cpu(float* y, const float* A, const float* x, uint32_t M, uint32_t N) {
  for (uint32_t i = 0; i < M; ++i) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
  }
}

static void show_usage() {
  std::cout << "HIP SGEMV (Matrix-Vector Multiplication)." << std::endl;
  std::cout << "Usage: [-m rows] [-n cols] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv, uint32_t &M, uint32_t &N) {
  int c;
  while ((c = getopt(argc, argv, "m:n:h")) != -1) {
    switch (c) {
    case 'm':
      M = atoi(optarg);
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t M = 1024;  // Rows (output vector size)
  uint32_t N = 1024;  // Columns (input vector size)

  // Parse command arguments
  parse_args(argc, argv, M, N);

  std::cout << "Matrix dimensions: " << M << " x " << N << std::endl;

  uint32_t A_size = M * N * sizeof(float);
  uint32_t x_size = N * sizeof(float);
  uint32_t y_size = M * sizeof(float);

  // Allocate host memory
  std::vector<float> h_A(M * N);
  std::vector<float> h_x(N);
  std::vector<float> h_y(M, 0.0f);
  std::vector<float> h_ref(M, 0.0f);

  // Generate synthetic data
  std::cout << "Generating test data" << std::endl;
  for (uint32_t i = 0; i < M * N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;  // Random matrix (0-1)
  }
  for (uint32_t i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;  // Random vector (0-1)
  }

  // Allocate device memory
  std::cout << "Allocating device memory" << std::endl;
  float *d_A = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;

  HIP_CHECK(hipMalloc(&d_A, A_size));
  HIP_CHECK(hipMalloc(&d_x, x_size));
  HIP_CHECK(hipMalloc(&d_y, y_size));

  // Copy input data to device
  std::cout << "Copying input data to device" << std::endl;
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size, hipMemcpyHostToDevice));

  // Launch kernel
  std::cout << "Launching kernel" << std::endl;
  uint32_t blockSize = 256;
  uint32_t gridSize = (M + blockSize - 1) / blockSize;

  auto time_start = std::chrono::high_resolution_clock::now();
  hipLaunchKernelGGL(kernel_sgemv, dim3(gridSize), dim3(blockSize), 0, 0,
                     d_A, d_x, d_y, M, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Copy results back to host
  std::cout << "Copying results from device" << std::endl;
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size, hipMemcpyDeviceToHost));

  // Verify results
  std::cout << "Verifying results" << std::endl;
  sgemv_cpu(h_ref.data(), h_A.data(), h_x.data(), M, N);

  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    if (fabs(h_y[i] - h_ref[i]) > 1e-3f) {
      if (errors < 10) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, h_ref[i], h_y[i]);
      }
      ++errors;
    }
  }

  // Cleanup
  std::cout << "Cleaning up" << std::endl;
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
