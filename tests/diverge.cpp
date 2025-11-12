#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <algorithm>

#define HIP_CHECK(cmd) do {                         \
    hipError_t error = cmd;                         \
    if (error != hipSuccess) {                      \
        printf("Error: '%s' returned %d: %s!\n", #cmd, (int)error, hipGetErrorString(error)); \
        exit(-1);                                   \
    }                                               \
} while(false)

///////////////////////////////////////////////////////////////////////////////

typedef struct {
  uint32_t num_points;
  int32_t* src_ptr;
  int32_t* dst_ptr;
} kernel_arg_t;

struct key_t {
    uint32_t user = 0;
};

///////////////////////////////////////////////////////////////////////////////

__device__ static void hacker(key_t* key, uint32_t task_id) {
    key->user = task_id;
}

__global__ void diverge_kernel(kernel_arg_t arg) {
    int32_t* src_ptr = arg.src_ptr;
    int32_t* dst_ptr = arg.dst_ptr;

    uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (task_id >= arg.num_points) return;

    int value = src_ptr[task_id];

    key_t key;
    uint32_t samples = arg.num_points;
    while (samples--) {
        hacker(&key, task_id);
        if ((key.user & 0x1) == 0) {
            value += 1;
        }
    }

    // none taken
    if (task_id >= 0x7fffffff) {
        value = 0;
    } else {
        value += 2;
    }

    // diverge
    if (task_id > 1) {
        if (task_id > 2) {
            value += 6;
        } else {
            value += 5;
        }
    } else {
        if (task_id > 0) {
            value += 4;
        } else {
            value += 3;
        }
    }

    // all taken
    if (task_id >= 0) {
        value += 7;
    } else {
        value = 0;
    }

    // loop
    for (int i = 0, n = task_id; i < n; ++i) {
        value += src_ptr[i];
    }

    // switch
    switch (task_id) {
    case 0:
        value += 1;
        break;
    case 1:
        value -= 1;
        break;
    case 2:
        value *= 3;
        break;
    case 3:
        value *= 5;
        break;
    default:
        break;
    }

    // select
    value += (task_id >= 0) ? ((task_id > 5) ? src_ptr[0] : task_id) : ((task_id < 5) ? src_ptr[1] : -task_id);

    // min/max
    value += min(src_ptr[task_id], value);
    value += max(src_ptr[task_id], value);

    dst_ptr[task_id] = value;
}

///////////////////////////////////////////////////////////////////////////////

void gen_src_data(std::vector<int32_t>& src_data, uint32_t size) {
    src_data.resize(size);
    for (uint32_t i = 0; i < size; ++i) {
        int32_t value = std::rand();
        src_data[i] = value;
    }
}

void gen_ref_data(std::vector<int32_t>& ref_data, const std::vector<int32_t>& src_data, uint32_t size) {
    ref_data.resize(size);
    for (int i = 0; i < (int)size; ++i) {
        int32_t value = src_data.at(i);

        uint32_t samples = size;
        while (samples--) {
            if ((i & 0x1) == 0) {
                value += 1;
            }
        }

        // none taken
        if (i >= 0x7fffffff) {
            value = 0;
        } else {
            value += 2;
        }

        // diverge
        if (i > 1) {
            if (i > 2) {
                value += 6;
            } else {
                value += 5;
            }
        } else {
            if (i > 0) {
                value += 4;
            } else {
                value += 3;
            }
        }

        // all taken
        if (i >= 0) {
            value += 7;
        } else {
            value = 0;
        }

        // loop
        for (int j = 0, n = i; j < n; ++j) {
            value += src_data.at(j);
        }

        // switch
        switch (i) {
        case 0:
            value += 1;
            break;
        case 1:
            value -= 1;
            break;
        case 2:
            value *= 3;
            break;
        case 3:
            value *= 5;
            break;
        default:
            assert(i < (int)size);
            break;
        }

        // select
        value += (i >= 0) ? ((i > 5) ? src_data.at(0) : i) : ((i < 5) ? src_data.at(1) : -i);

        // min/max
        value += std::min(src_data.at(i), value);
        value += std::max(src_data.at(i), value);

        ref_data[i] = value;
    }
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    uint32_t count = 1;

    // parse command arguments
    int c;
    while ((c = getopt(argc, argv, "n:h")) != -1) {
        switch (c) {
        case 'n':
            count = atoi(optarg);
            break;
        case 'h':
            std::cout << "HIP Diverge Test." << std::endl;
            std::cout << "Usage: [-n count] [-h: help]" << std::endl;
            exit(0);
            break;
        default:
            std::cout << "HIP Diverge Test." << std::endl;
            std::cout << "Usage: [-n count] [-h: help]" << std::endl;
            exit(-1);
        }
    }

    if (count == 0) {
        count = 1;
    }

    std::srand(50);

    // Get device properties
    int device = 0;
    hipDeviceProp_t device_prop;
    HIP_CHECK(hipGetDeviceProperties(&device_prop, device));
    HIP_CHECK(hipSetDevice(device));

    std::cout << "Device: " << device_prop.name << std::endl;

    // Use device max threads per block as block size
    uint32_t block_size = 256;
    uint32_t num_points = count * device_prop.multiProcessorCount * 8;  // estimate total work
    uint32_t grid_size = (num_points + block_size - 1) / block_size;
    uint32_t buf_size = num_points * sizeof(int32_t);

    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;
    std::cout << "grid size: " << grid_size << ", block size: " << block_size << std::endl;

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    int32_t* d_src = nullptr;
    int32_t* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, buf_size));
    HIP_CHECK(hipMalloc(&d_dst, buf_size));

    // allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<int32_t> h_src;
    std::vector<int32_t> h_dst(num_points);
    gen_src_data(h_src, num_points);

    // upload source buffer
    std::cout << "upload source buffer" << std::endl;
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), buf_size, hipMemcpyHostToDevice));

    // setup kernel arguments
    std::cout << "setup kernel arguments" << std::endl;
    kernel_arg_t kernel_arg;
    kernel_arg.num_points = num_points;
    kernel_arg.src_ptr = d_src;
    kernel_arg.dst_ptr = d_dst;

    // launch kernel
    std::cout << "launch kernel" << std::endl;
    hipLaunchKernelGGL(diverge_kernel,
                       dim3(grid_size),
                       dim3(block_size),
                       0, 0,
                       kernel_arg);
    HIP_CHECK(hipGetLastError());

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, buf_size, hipMemcpyDeviceToHost));

    // verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<int32_t> h_ref;
        gen_ref_data(h_ref, h_src, num_points);

        for (uint32_t i = 0; i < num_points; ++i) {
            int32_t ref = h_ref[i];
            int32_t cur = h_dst[i];
            if (cur != ref) {
                std::cout << "error at result #" << std::dec << i
                          << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
                ++errors;
            }
        }
    }

    // cleanup
    std::cout << "cleanup" << std::endl;
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
