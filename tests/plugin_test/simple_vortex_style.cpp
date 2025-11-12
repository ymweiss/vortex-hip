// Simple kernel in Vortex C++ style
// This will compile to regular C++ and we can see the IR

#include <stdint.h>

// Vortex-style dim3
typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t m[3];
} dim3_t;

// Thread-local built-ins (like Vortex)
extern __thread dim3_t blockIdx;
extern __thread dim3_t threadIdx;
extern dim3_t gridDim;
extern dim3_t blockDim;

// Simple vecadd kernel function
// We want to mark this as a HIP kernel somehow
void vecadd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Mixed types kernel for testing alignment
void mixed_types(
    char c,
    short s,
    int i,
    long l,
    float* ptr1,
    double* ptr2,
    int final
) {
    // Just a test for alignment
}
