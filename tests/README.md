# Vortex HIP Test Suite

This directory contains HIP test programs migrated from the Vortex repository (`vortex/hip/`).

## Test Programs

These are complete HIP applications that test various features of the HIP runtime on Vortex:

### Basic Tests
- **vecadd.cpp** - Vector addition (basic parallel operation)
- **basic.cpp** - Memory copy operations and basic kernel launch
- **fence.cpp** - Memory fence and synchronization primitives
- **printf.cpp** - Device-side printf functionality

### Math & Linear Algebra
- **dotproduct.cpp** - Dot product computation
- **sgemm.cpp** - Single-precision general matrix multiply (basic)
- **sgemm2.cpp** - SGEMM with shared memory tiling
- **sgemm_tcu.cpp** - SGEMM using Tensor Compute Unit (TCU)
- **sgemv.cpp** - Single-precision general matrix-vector multiply

### Neural Network Operations
- **relu.cpp** - ReLU activation function
- **dropout.cpp** - Dropout layer
- **conv3.cpp** - 3D convolution

### Advanced Tests
- **diverge.cpp** - Thread divergence and control flow
- **cta.cpp** - Cooperative Thread Array (CTA) operations
- **sort.cpp** - Parallel sorting algorithms
- **stencil3d.cpp** - 3D stencil computation
- **madmax.cpp** - Memory access patterns and optimization
- **mstress.cpp** - Memory stress testing
- **io_addr.cpp** - I/O and address space handling

### Comprehensive Tests
- **demo.cpp** - Demonstration of multiple HIP features
- **dogfood.cpp** - Comprehensive functionality test ("eating our own dog food")

## Building Tests

### Prerequisites

1. **Vortex GPU** installed and built at `~/vortex`
2. **HIP Runtime** for Vortex (this repository's runtime library)
3. **hipcc** or Clang with HIP support configured for Vortex target

### Build Instructions

**Note:** These tests require a HIP compiler configured to target Vortex. The build system for these tests is currently under development.

#### Manual Compilation Example

```bash
# Set environment
export VORTEX_ROOT=~/vortex
export HIP_PATH=/path/to/hip

# Compile with hipcc (when available)
hipcc vecadd.cpp \
    -I../runtime/include \
    -L../runtime/build \
    -lhip_vortex \
    -o vecadd

# Run on Vortex
export LD_LIBRARY_PATH=$VORTEX_ROOT/build/runtime:../runtime/build:$LD_LIBRARY_PATH
./vecadd
```

## Test Structure

Each test follows a similar pattern:

```cpp
#include <hip/hip_runtime.h>

// HIP kernel
__global__ void kernel(...) {
    // Kernel code
}

int main() {
    // 1. Allocate device memory
    hipMalloc(...);

    // 2. Copy data to device
    hipMemcpy(..., hipMemcpyHostToDevice);

    // 3. Launch kernel
    kernel<<<grid, block>>>(...);

    // 4. Copy results back
    hipMemcpy(..., hipMemcpyDeviceToHost);

    // 5. Verify results
    // 6. Cleanup
    hipFree(...);
}
```

## Features Tested

### Memory Operations
- `hipMalloc` / `hipFree` - Device memory allocation
- `hipMemcpy` - Host-device data transfer
- `hipMemset` - Device memory initialization

### Kernel Launch
- `<<<grid, block>>>` - Kernel launch syntax
- Grid and block dimensions
- Shared memory allocation
- Multiple kernels

### Synchronization
- `hipDeviceSynchronize()` - Host-device sync
- `__syncthreads()` - Thread block synchronization
- Memory fences

### Device Functions
- Thread indexing (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`)
- Warp primitives (shuffles, voting)
- Atomic operations
- Math functions

### Advanced Features
- Cooperative groups
- Dynamic shared memory
- Device-side printf
- Texture/TCU operations

## Running Tests

**Status:** These tests are reference implementations. Full integration with the Vortex HIP runtime is in progress.

Current work items:
- [ ] Implement complete argument marshaling
- [ ] Set up CMake build system for tests
- [ ] Configure hipcc for Vortex target
- [ ] Validate each test on Vortex hardware/simulator
- [ ] Create automated test runner

## Notes

- Original source: `vortex/hip/` directory in Vortex GPU repository
- These tests were originally written for the Vortex GPU and demonstrate real-world HIP usage patterns
- Some tests may require specific Vortex hardware features (e.g., TCU for `sgemm_tcu.cpp`)
- Tests use `HIP_CHECK()` macro for error checking

## Contributing

When adding new tests:
1. Follow the existing structure and naming conventions
2. Include error checking with `HIP_CHECK()`
3. Add verification of results
4. Document special requirements (hardware features, memory size, etc.)
5. Update this README with test description
