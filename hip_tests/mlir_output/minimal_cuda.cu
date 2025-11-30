// Minimal CUDA kernel to demonstrate Polygeist --cuda-lower output
// Using .cu extension so Polygeist recognizes it as CUDA code

extern "C" {

// Minimal CUDA built-in declarations
struct uint3 {
    unsigned int x, y, z;
};

struct dim3 {
    unsigned int x, y, z;
};

__attribute__((device)) uint3 threadIdx;
__attribute__((device)) uint3 blockIdx;
__attribute__((device)) dim3 blockDim;
__attribute__((device)) dim3 gridDim;

// Simple kernel
__attribute__((global)) void simple_add(int* data, int value) {
    int idx = threadIdx.x;
    data[idx] = data[idx] + value;
}

}

int main() {
    return 0;
}
