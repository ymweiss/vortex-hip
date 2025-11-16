// Minimal HIP kernel to test Polygeist's --cuda-lower output
// This is intentionally simple to see clean GPU dialect IR

__global__ void simple_add(int* data, int value) {
    int idx = threadIdx.x;
    data[idx] = data[idx] + value;
}

int main() {
    return 0;
}
