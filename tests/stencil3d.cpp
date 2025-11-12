#include <hip/hip_runtime.h>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "error: '%s' returned %d (%s)!\n", #cmd, error, hipGetErrorString(error)); \
            cleanup(); \
            exit(-1); \
        } \
    } while (false)

#define FLOAT_ULP 6

// Data type (can be overridden at compile time)
#ifndef TYPE
#define TYPE float
#endif

///////////////////////////////////////////////////////////////////////////////
// Comparator templates

template <typename Type>
class Comparator
{
};

template <>
class Comparator<int>
{
public:
    static const char *type_str()
    {
        return "integer";
    }
    static int generate()
    {
        return rand();
    }
    static bool compare(int a, int b, int index, int errors)
    {
        if (a != b)
        {
            if (errors < 100)
            {
                printf("*** error: [%d] expected=%d, actual=%d\n", index, a, b);
            }
            return false;
        }
        return true;
    }
};

template <>
class Comparator<float>
{
private:
    union Float_t
    {
        float f;
        int i;
    };

public:
    static const char *type_str()
    {
        return "float";
    }
    static float generate()
    {
        return static_cast<float>(rand()) / RAND_MAX;
    }
    static bool compare(float a, float b, int index, int errors)
    {
        union fi_t
        {
            float f;
            int32_t i;
        };
        fi_t fa, fb;
        fa.f = a;
        fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP)
        {
            if (errors < 100)
            {
                printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
            }
            return false;
        }
        return true;
    }
};

///////////////////////////////////////////////////////////////////////////////
// HIP Kernel

__global__ void stencil3d_kernel(TYPE *A, TYPE *B, uint32_t size)
{
    // Calculate global column, row, and depth indices using both block and thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds check
    if (col >= size || row >= size || dep >= size)
        return;

    TYPE sum = 0;
    int count = 0;

    // Stencil kernel size is 3x3x3
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                // Compute the neighbor's index, handling boundary conditions
                int nz = dep + dz;
                int ny = row + dy;
                int nx = col + dx;

                // Clamp the indices to be within the boundary of the array
                if (nz < 0) {
                    nz = 0;
                } else if (nz >= (int)size) {
                    nz = size - 1;
                }
                if (ny < 0) {
                    ny = 0;
                } else if (ny >= (int)size) {
                    ny = size - 1;
                }
                if (nx < 0) {
                    nx = 0;
                } else if (nx >= (int)size) {
                    nx = size - 1;
                }

                // Add the neighbor's value to sum
                sum += A[nz * size * size + ny * size + nx];
                count++;
            }
        }
    }

    // Compute the average of the sum of neighbors and write to the output array
    B[dep * size * size + row * size + col] = sum / count;
}

///////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation

static void stencil_cpu(TYPE *out, const TYPE *in, uint32_t width, uint32_t height, uint32_t depth)
{
    // Handle boundary conditions using boundary replication
    for (uint32_t z = 0; z < depth; z++)
    {
        for (uint32_t y = 0; y < height; y++)
        {
            for (uint32_t x = 0; x < width; x++)
            {
                TYPE sum = 0;
                int count = 0;

                // Iterate over the neighborhood
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            // Compute the neighbor's index
                            int nx = (int)x + dx;
                            int ny = (int)y + dy;
                            int nz = (int)z + dz;

                            // Check bounds and replicate the boundary values
                            if (nx < 0) {
                                nx = 0;
                            } else if (nx >= (int)width) {
                                nx = width - 1;
                            }

                            if (ny < 0) {
                                ny = 0;
                            } else if (ny >= (int)height) {
                                ny = height - 1;
                            }

                            if (nz < 0) {
                                nz = 0;
                            } else if (nz >= (int)depth) {
                                nz = depth - 1;
                            }

                            // Sum up the values
                            sum += in[nz * width * height + ny * width + nx];
                            count++;
                        }
                    }
                }

                // Write the averaged value to the output array
                out[z * width * height + y * width + x] = sum / count;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Global variables

uint32_t size = 64;
uint32_t block_size = 2;

TYPE *d_A = nullptr;
TYPE *d_B = nullptr;

static void show_usage()
{
    std::cout << "HIP Stencil3D Test." << std::endl;
    std::cout << "Usage: [-n matrix_size] [-b:block_size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "n:b:h")) != -1)
    {
        switch (c)
        {
        case 'n':
            size = atoi(optarg);
            break;
        case 'b':
            block_size = atoi(optarg);
            break;
        case 'h':
            show_usage();
            exit(0);
            break;
        default:
            show_usage();
            exit(-1);
        }
    }
}

void cleanup()
{
    if (d_A)
    {
        HIP_CHECK(hipFree(d_A));
    }
    if (d_B)
    {
        HIP_CHECK(hipFree(d_B));
    }
}

int main(int argc, char *argv[])
{
    // Parse command arguments
    parse_args(argc, argv);

    if ((size / block_size) * block_size != size)
    {
        printf("Error: matrix size %d must be a multiple of block size %d\n", size, block_size);
        return -1;
    }

    std::srand(50);

    // Check for available HIP devices
    std::cout << "Checking HIP devices..." << std::endl;
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "Error: No HIP devices found!" << std::endl;
        return -1;
    }
    std::cout << "Found " << device_count << " HIP device(s)" << std::endl;

    uint32_t size_cubed = size * size * size;
    uint32_t buf_size = size_cubed * sizeof(TYPE);

    std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
    std::cout << "matrix size: " << size << "x" << size << "x" << size << std::endl;
    std::cout << "block size: " << block_size << "x" << block_size << "x" << block_size << std::endl;

    // Allocate device memory
    std::cout << "allocate device memory" << std::endl;
    HIP_CHECK(hipMalloc(&d_A, buf_size));
    HIP_CHECK(hipMalloc(&d_B, buf_size));

    // Allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<TYPE> h_A(size_cubed);
    std::vector<TYPE> h_B(size_cubed);

    // Generate source data
    for (uint32_t i = 0; i < size_cubed; ++i)
    {
        h_A[i] = Comparator<TYPE>::generate();
    }

    // Copy input data to device
    std::cout << "copy input data to device" << std::endl;
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), buf_size, hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "launch kernel" << std::endl;
    dim3 grid((size + block_size - 1) / block_size,
              (size + block_size - 1) / block_size,
              (size + block_size - 1) / block_size);
    dim3 block(block_size, block_size, block_size);

    hipLaunchKernelGGL(stencil3d_kernel, grid, block, 0, 0, d_A, d_B, size);
    HIP_CHECK(hipGetLastError());

    // Wait for kernel to complete
    std::cout << "wait for kernel completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output data from device
    std::cout << "copy output data from device" << std::endl;
    HIP_CHECK(hipMemcpy(h_B.data(), d_B, buf_size, hipMemcpyDeviceToHost));

    // Verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<TYPE> h_ref(size_cubed);
        stencil_cpu(h_ref.data(), h_A.data(), size, size, size);

        for (uint32_t i = 0; i < h_ref.size(); ++i)
        {
            if (!Comparator<TYPE>::compare(h_B[i], h_ref[i], i, errors))
            {
                ++errors;
            }
        }
    }

    // Cleanup
    std::cout << "cleanup" << std::endl;
    cleanup();

    if (errors != 0)
    {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
