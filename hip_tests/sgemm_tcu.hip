// SGEMM_TCU (Tensor Core Unit Matrix Multiplication) Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/sgemm_tcu
// Computes C = A * B for matrices using tensor-core-style tiled operations
//
// Note: This is a simplified version that uses standard operations since
// Vortex may not have actual HIP tensor cores. The tiling pattern follows
// the original TCU test structure.

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>

// Default matrix element types
#ifndef ITYPE
#define ITYPE float
#endif

#ifndef OTYPE
#define OTYPE float
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

// Tensor Core Unit configuration
// These define the tile sizes for matrix multiplication
#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 16
#endif

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for tensor-core-style tiled matrix multiplication
// This kernel processes one tile of the output matrix C
__global__ void sgemm_tcu_kernel(ITYPE* A, ITYPE* B, OTYPE* C,
                                  uint32_t M, uint32_t N, uint32_t K) {
  // Calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * TILE_M;
  uint32_t tile_col = blockIdx.x * TILE_N;

  // Each thread computes one element within the tile
  uint32_t row = tile_row + threadIdx.y;
  uint32_t col = tile_col + threadIdx.x;

  if (row < M && col < N) {
    OTYPE sum = 0;

    // Iterate over tiles in the K dimension
    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
      // Accumulate over elements in this K-tile
      for (uint32_t e = 0; e < TILE_K && (k_tile + e) < K; ++e) {
        uint32_t k = k_tile + e;
        ITYPE a_val = A[row * K + k];
        ITYPE b_val = B[k * N + col];
        sum += static_cast<OTYPE>(a_val) * static_cast<OTYPE>(b_val);
      }
    }

    C[row * N + col] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

static void show_usage() {
  printf("Vortex SGEMM TCU Test - HIP version\n");
  printf("Usage: [-m M] [-n N] [-k K] [-h: help]\n");
  printf("  M, N, K should be multiples of tile dimensions (16x16x16)\n");
}

int main(int argc, char *argv[]) {
  uint32_t M = 32;
  uint32_t N = 32;
  uint32_t K = 32;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      M = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      N = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      K = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-h") == 0) {
      show_usage();
      return 0;
    }
  }

  // Validate dimensions are multiples of tile sizes
  if ((M % TILE_M) != 0) {
    std::cerr << "Error: M (" << M << ") must be a multiple of TILE_M (" << TILE_M << ")" << std::endl;
    return -1;
  }
  if ((N % TILE_N) != 0) {
    std::cerr << "Error: N (" << N << ") must be a multiple of TILE_N (" << TILE_N << ")" << std::endl;
    return -1;
  }
  if ((K % TILE_K) != 0) {
    std::cerr << "Error: K (" << K << ") must be a multiple of TILE_K (" << TILE_K << ")" << std::endl;
    return -1;
  }

  std::srand(50);

  uint32_t size_A = M * K;
  uint32_t size_B = K * N;
  uint32_t size_C = M * N;

  uint32_t buf_size_A = size_A * sizeof(ITYPE);
  uint32_t buf_size_B = size_B * sizeof(ITYPE);
  uint32_t buf_size_C = size_C * sizeof(OTYPE);

  std::cout << "Input data type: " << typeid(ITYPE).name() << std::endl;
  std::cout << "Output data type: " << typeid(OTYPE).name() << std::endl;
  std::cout << "Tile Dimension: M=" << TILE_M << ", N=" << TILE_N << ", K=" << TILE_K << std::endl;
  std::cout << "Matrix A: " << M << "x" << K << std::endl;
  std::cout << "Matrix B: " << K << "x" << N << std::endl;
  std::cout << "Matrix C: " << M << "x" << N << std::endl;
  std::cout << "Buffer sizes: A=" << buf_size_A << " bytes, B=" << buf_size_B
            << " bytes, C=" << buf_size_C << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<ITYPE> h_A(size_A);
  std::vector<ITYPE> h_B(size_B);
  std::vector<OTYPE> h_C(size_C);

  // Initialize matrices with random values
  printf("Initialize matrices\n");
  for (uint32_t i = 0; i < size_A; ++i) {
    h_A[i] = static_cast<ITYPE>(rand()) / RAND_MAX;
  }
  for (uint32_t i = 0; i < size_B; ++i) {
    h_B[i] = static_cast<ITYPE>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  printf("Allocate device memory\n");
  ITYPE *d_A, *d_B;
  OTYPE *d_C;
  HIP_CHECK(hipMalloc(&d_A, buf_size_A));
  HIP_CHECK(hipMalloc(&d_B, buf_size_B));
  HIP_CHECK(hipMalloc(&d_C, buf_size_C));

  // Upload matrices to device
  printf("Upload matrix A buffer\n");
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size_A, hipMemcpyHostToDevice));

  printf("Upload matrix B buffer\n");
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size_B, hipMemcpyHostToDevice));

  // Configure kernel launch parameters
  // Grid dimensions based on tiles
  dim3 threads(TILE_N, TILE_M);
  dim3 blocks(N / TILE_N, M / TILE_M);

  std::cout << "Launch kernel with grid(" << blocks.x << "," << blocks.y
            << ") threads(" << threads.x << "," << threads.y << ")" << std::endl;

  // Launch kernel
  hipLaunchKernelGGL(sgemm_tcu_kernel, blocks, threads, 0, 0,
                     d_A, d_B, d_C, M, N, K);

  // Wait for completion
  printf("Wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  printf("Download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size_C, hipMemcpyDeviceToHost));

  // Verify result with CPU computation
  printf("Verify result\n");
  int errors = 0;
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      OTYPE sum = 0;
      for (uint32_t k = 0; k < K; ++k) {
        ITYPE a_val = h_A[row * K + k];
        ITYPE b_val = h_B[k * N + col];
        sum += static_cast<OTYPE>(a_val) * static_cast<OTYPE>(b_val);
      }
      OTYPE cur = h_C[row * N + col];
      OTYPE diff = std::abs(cur - sum);

      // Use ULP-based comparison for floating point
      OTYPE tolerance = FLOAT_ULP * std::numeric_limits<OTYPE>::epsilon() * K;
      if (diff > tolerance) {
        if (errors < 100) {
          printf("*** error: [%d,%d] expected=%f, actual=%f, diff=%f\n",
                 row, col, sum, cur, diff);
        }
        ++errors;
      }
    }
  }

  // Cleanup
  printf("Cleanup\n");
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << errors << " / " << size_C << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
