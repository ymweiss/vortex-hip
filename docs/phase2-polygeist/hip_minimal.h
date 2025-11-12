// Minimal HIP header for testing Polygeist compatibility
// HIP is CUDA-compatible by design

#ifndef HIP_MINIMAL_H
#define HIP_MINIMAL_H

// HIP attributes (identical to CUDA)
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))

// dim3 structure
struct dim3 {
    unsigned int x, y, z;

    __host__ __device__ dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// HIP built-in variables (same as CUDA)
extern const dim3 threadIdx;
extern const dim3 blockIdx;
extern const dim3 blockDim;
extern const dim3 gridDim;

// HIP launch bounds (same as CUDA)
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

#endif // HIP_MINIMAL_H
