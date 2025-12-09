// SGEMM2 (Tiled Matrix Multiplication) Test - HIP version
// Converted from Vortex regression test: vortex/tests/regression/sgemm2
// Computes C = A * B for square matrices using shared memory tiling

#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <stdio.h>

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

// HIP kernel for tiled matrix multiplication using shared memory
__global__ void sgemm2_kernel(TYPE* A, TYPE* B, TYPE* C, uint32_t size, uint32_t tile_size) {
  // Determine global row and column indices
  int g_row = blockIdx.x * blockDim.x + threadIdx.x;
  int g_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Determine local row and column indices
  int l_row = threadIdx.x;
  int l_col = threadIdx.y;

  // Allocate shared memory for tiles of A and B
  extern __shared__ TYPE local_mem[];
  TYPE* local_A = local_mem;
  TYPE* local_B = local_mem + blockDim.x * blockDim.y;

  TYPE sum = 0;

  // Loop over tiles
  for (uint32_t k = 0; k < size; k += tile_size) {
    // Load tile of matrix A & B to shared memory
    if (g_row < size && (k + l_col) < size) {
      local_A[l_row * tile_size + l_col] = A[g_row * size + (k + l_col)];
    } else {
      local_A[l_row * tile_size + l_col] = 0;
    }

    if ((k + l_row) < size && g_col < size) {
      local_B[l_row * tile_size + l_col] = B[(k + l_row) * size + g_col];
    } else {
      local_B[l_row * tile_size + l_col] = 0;
    }

    // Synchronize all threads in current block
    __syncthreads();

    // Compute partial sum for the local tile
    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
    }

    // Synchronize all threads in current block
    __syncthreads();
  }

  // Store the computed sum into the result matrix C
  if (g_row < size && g_col < size) {
    C[g_row * size + g_col] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////

// CPU reference implementation
void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum = 0;
      for (uint32_t e = 0; e < width; ++e) {
        TYPE a = A[row * width + e];
        TYPE b = B[e * width + col];
        TYPE c = a * b;
        sum += c;
      }
      out[row * width + col] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 16;
  uint32_t tile_size = 4;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      tile_size = atoi(argv[i + 1]);
      i++;
    }
  }

  if ((size / tile_size) * tile_size != size) {
    std::cerr << "Error: matrix size " << size << " must be a multiple of tile size " << tile_size << std::endl;
    return 1;
  }

  std::srand(50);

  uint32_t num_elements = size * size;
  uint32_t buf_size = num_elements * sizeof(TYPE);
  uint32_t shared_mem_size = 2 * tile_size * tile_size * sizeof(TYPE);

  printf("data type: float\n");
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "tile size: " << tile_size << "x" << tile_size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;
  std::cout << "shared memory: " << shared_mem_size << " bytes" << std::endl;

  // Allocate host buffers
  std::vector<TYPE> h_A(num_elements);
  std::vector<TYPE> h_B(num_elements);
  std::vector<TYPE> h_C(num_elements);

  // Initialize matrices with random values
  for (uint32_t i = 0; i < num_elements; ++i) {
    h_A[i] = static_cast<TYPE>(rand()) / RAND_MAX;
    h_B[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  TYPE *d_A, *d_B, *d_C;
  HIP_CHECK(hipMalloc(&d_A, buf_size));
  HIP_CHECK(hipMalloc(&d_B, buf_size));
  HIP_CHECK(hipMalloc(&d_C, buf_size));

  // Upload matrices
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  dim3 threads(tile_size, tile_size);
  dim3 blocks((size + tile_size - 1) / tile_size, (size + tile_size - 1) / tile_size);

  std::cout << "launch kernel with grid(" << blocks.x << "," << blocks.y
            << ") block(" << threads.x << "," << threads.y << ")" << std::endl;

  hipLaunchKernelGGL(sgemm2_kernel, blocks, threads, shared_mem_size, 0,
                     d_A, d_B, d_C, size, tile_size);

  // Wait for completion
  HIP_CHECK(hipDeviceSynchronize());

  // Download result
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, buf_size, hipMemcpyDeviceToHost));

  // Verify result with CPU computation
  printf("verify result\n");
  int errors = 0;
  {
    std::vector<TYPE> h_ref(num_elements);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      TYPE expected = h_ref[i];
      TYPE actual = h_C[i];

      // Use ULP-based comparison for floating point
      union fi_t { TYPE f; int32_t i; };
      fi_t fa, fb;
      fa.f = expected;
      fb.f = actual;
      auto d = std::abs(fa.i - fb.i);

      if (d > FLOAT_ULP) {
        if (errors < 100) {
          uint32_t row = i / size;
          uint32_t col = i % size;
          printf("*** error: [%d,%d] expected=%f, actual=%f\n", row, col, expected, actual);
        }
        ++errors;
      }
    }
  }

  // Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
