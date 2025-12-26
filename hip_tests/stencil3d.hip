// 3D Stencil Computation - HIP version
// Converted from Vortex regression test: vortex/tests/regression/stencil3d

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
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, (double)b, (double)a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

// HIP kernel for 3D stencil computation
__global__ void stencil3d_kernel(TYPE* in, TYPE* out, uint32_t size) {
  // Calculate global column, row, and depth indices using both block and thread indices
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int dep = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if within bounds
  if (col >= size || row >= size || dep >= size) {
    return;
  }

  TYPE sum = 0;
  int count = 0;

  // Stencil kernel size is assumed to be 3x3x3
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        // Compute the neighbor's index, handling boundary conditions manually
        int nz = dep + dz;
        int ny = row + dy;
        int nx = col + dx;

        // Clamp the indices to be within the boundary of the array
        if (nz < 0) {
          nz = 0;
        } else if (nz >= size) {
          nz = size - 1;
        }
        if (ny < 0) {
          ny = 0;
        } else if (ny >= size) {
          ny = size - 1;
        }
        if (nx < 0) {
          nx = 0;
        } else if (nx >= size) {
          nx = size - 1;
        }

        // Add the neighbor's value to sum
        sum += in[nz * size * size + ny * size + nx];
        count++;
      }
    }
  }

  // Compute the average of the sum of neighbors and write to the output array
  out[dep * size * size + row * size + col] = sum / count;
}

///////////////////////////////////////////////////////////////////////////////

// CPU reference implementation
static void stencil3d_cpu(TYPE *out, const TYPE *in, uint32_t size) {
  // We'll need to handle boundary conditions. Let's assume we use boundary replication.
  for (uint32_t z = 0; z < size; z++) {
    for (uint32_t y = 0; y < size; y++) {
      for (uint32_t x = 0; x < size; x++) {
        TYPE sum = 0;
        int count = 0;

        // Iterate over the neighborhood
        for (int dz = -1; dz <= 1; dz++) {
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              // Compute the neighbor's index
              int nx = (int)x + dx;
              int ny = (int)y + dy;
              int nz = (int)z + dz;

              // Check bounds and replicate the boundary values
              if (nx < 0) {
                nx = 0;
              } else if (nx >= (int)size) {
                nx = size - 1;
              }

              if (ny < 0) {
                ny = 0;
              } else if (ny >= (int)size) {
                ny = size - 1;
              }

              if (nz < 0) {
                nz = 0;
              } else if (nz >= (int)size) {
                nz = size - 1;
              }

              // Sum up the values
              sum += in[nz * size * size + ny * size + nx];
              count++;
            }
          }
        }

        // Write the averaged value to the output array
        out[z * size * size + y * size + x] = sum / count;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  uint32_t size = 8;
  uint32_t block_size = 2;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      size = atoi(argv[i + 1]);
      i++;
    } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
      block_size = atoi(argv[i + 1]);
      i++;
    }
  }

  // Validate that size is a multiple of block_size
  if ((size / block_size) * block_size != size) {
    std::cerr << "Error: matrix size " << size << " must be a multiple of block size " << block_size << std::endl;
    return 1;
  }

  std::srand(50);

  uint32_t size_cubed = size * size * size;
  uint32_t buf_size = size_cubed * sizeof(TYPE);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << "x" << size << std::endl;
  std::cout << "block size: " << block_size << "x" << block_size << "x" << block_size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // Allocate host buffers
  printf("allocate host buffers\n");
  std::vector<TYPE> h_in(size_cubed);
  std::vector<TYPE> h_out(size_cubed);

  // Generate source data
  for (uint32_t i = 0; i < size_cubed; ++i) {
    h_in[i] = Comparator<TYPE>::generate();
  }

  // Allocate device memory
  printf("allocate device memory\n");
  TYPE *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, buf_size));
  HIP_CHECK(hipMalloc(&d_out, buf_size));

  // Upload source buffer
  printf("upload source buffer\n");
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), buf_size, hipMemcpyHostToDevice));

  // Launch kernel
  printf("launch kernel\n");
  dim3 threads_per_block(block_size, block_size, block_size);
  dim3 num_blocks(size / block_size, size / block_size, size / block_size);

  hipLaunchKernelGGL(stencil3d_kernel, num_blocks, threads_per_block, 0, 0,
                     d_in, d_out, size);

  // Wait for completion
  printf("wait for completion\n");
  HIP_CHECK(hipDeviceSynchronize());

  // Download destination buffer
  printf("download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, buf_size, hipMemcpyDeviceToHost));

  // Verify result
  printf("verify result\n");
  int errors = 0;
  {
    std::vector<TYPE> h_ref(size_cubed);
    stencil3d_cpu(h_ref.data(), h_in.data(), size);

    for (uint32_t i = 0; i < size_cubed; ++i) {
      if (!Comparator<TYPE>::compare(h_out[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // Cleanup
  printf("cleanup\n");
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
