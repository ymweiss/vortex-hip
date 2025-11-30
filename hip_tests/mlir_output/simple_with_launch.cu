// Simple CUDA kernel WITH launch to see GPU dialect output

__global__ void simple_add(int* data, int value) {
    int idx = threadIdx.x;
    data[idx] = data[idx] + value;
}

int main() {
    int* d_data = 0;

    // Launch kernel
    simple_add<<<1, 256>>>(d_data, 42);

    return 0;
}
