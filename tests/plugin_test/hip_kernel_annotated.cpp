// HIP-style kernel using Vortex annotation approach
// Define __global__ as an annotation that LLVM can detect

#include <stdint.h>

// HIP kernel marker (like Vortex's __UNIFORM__)
#define __global__ __attribute__((annotate("hip.kernel")))

// Vortex-style dim3
typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t m[3];
} dim3_t;

// Thread-local built-ins
extern __thread dim3_t blockIdx;
extern __thread dim3_t threadIdx;
extern dim3_t gridDim;
extern dim3_t blockDim;

// HIP kernel: vecadd
__global__ void vecadd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// HIP kernel: mixed_types (for alignment testing)
__global__ void mixed_types(
    char c,
    short s,
    int i,
    long l,
    float* ptr1,
    double* ptr2,
    int final
) {
    // Test alignment
}

// Regular function (not a kernel)
void helper_function(int x) {
    // This should NOT be detected as a kernel
}
