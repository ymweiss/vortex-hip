# Phase 1: HIP Runtime API Testing

**Status:** ✅ In Progress
**Purpose:** Verify HIP runtime API correctly maps to Vortex API calls

## Overview

Phase 1 tests verify that the HIP runtime library correctly implements the HIP API by mapping calls to the underlying Vortex API. These tests use **manually written Vortex kernels** - this is intentional and correct for Phase 1.

## What Phase 1 Tests

### HIP API → Vortex API Mapping
- `hipSetDevice()` → `vx_dev_open()`
- `hipGetDeviceProperties()` → `vx_dev_caps()`
- `hipMalloc()` → `vx_mem_alloc()`
- `hipFree()` → `vx_mem_free()`
- `hipMemcpy()` → `vx_copy_to_dev()` / `vx_copy_from_dev()`
- `hipLaunchKernel()` → `vx_upload_kernel_bytes()` + `vx_start()`
- `hipDeviceSynchronize()` → `vx_ready_wait()`

### Metadata-Driven Argument Marshaling
- Convert HIP's array-of-pointers calling convention
- Pack into Vortex's struct format
- Handle RV32 architecture (4-byte pointers)
- Use metadata from DWARF debug info

## Test Organization

All tests are located in `tests/` with the following structure:

```
tests/
├── basic_test/           # Basic device/memory operations
├── vecadd_test/          # Vector addition
├── sgemm_test/           # Matrix multiplication
├── dotproduct_test/      # Dot product
├── relu_test/            # ReLU activation
├── fence_test/           # Memory fences
├── conv3_test/           # 3D convolution
├── cta_test/             # Cooperative thread arrays
├── sgemm2_test/          # Shared memory tiling
├── diverge_test/         # Control flow divergence
├── madmax_test/          # Computational stress
├── mstress_test/         # Memory stress
└── demo_test/            # Comprehensive demo
```

## Why Manually Written Kernels?

**Phase 1 Goal:** Test the runtime API mapping, not compilation.

- Kernels are written directly in Vortex format using `vx_spawn.h`
- This isolates runtime testing from compiler concerns
- Compilation from HIP `__global__` kernels is Phase 2 work

**This is correct!** Phase 1 should verify the runtime works independently.

## Test Pattern

Each test follows this structure:

### 1. Kernel (kernel.cpp)
```cpp
#include <vx_spawn.h>

struct KernelArgs {
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint64_t shared_mem;
    // User arguments
} __attribute__((packed));

void kernel_body(KernelArgs* __UNIFORM__ args) {
    // Kernel implementation using Vortex intrinsics
}

int main() {
    KernelArgs* args = (KernelArgs*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(dim, args->grid_dim, args->block_dim,
                           (vx_kernel_func_cb)kernel_body, args);
}
```

### 2. Host Code (main.cpp)
```cpp
#include "vortex_hip_runtime.h"

extern void* kernel_body_handle;

int main() {
    // Use HIP API
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_ptr, size));
    HIP_CHECK(hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice));

    void* args[] = {&d_ptr, &param1, &param2};
    HIP_CHECK(hipLaunchKernel(kernel_body_handle, grid, block,
                             args, 0, nullptr));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_result, d_ptr, size, hipMemcpyDeviceToHost));

    // Verify results
}
```

### 3. Build Process
```makefile
# Compile kernel to RISC-V
kernel.elf: kernel.cpp
    $(RISCV_CXX) -g kernel.cpp -o kernel.elf

# Extract metadata from DWARF
kernel_metadata.cpp: kernel.elf
    python3 hip_metadata_gen.py kernel.elf > kernel_metadata.cpp

# Convert to Vortex binary
kernel.vxbin: kernel.elf
    vxbin.py kernel.elf kernel.vxbin

# Link everything
test: main.o kernel_metadata.o kernel_vxbin.o
    $(CXX) main.o kernel_metadata.o kernel_vxbin.o -lhip_vortex -o test
```

## Test Categories

### Phase 1A: Basic Runtime (✅ Complete)
- Device management
- Memory allocation
- Data transfer
- Kernel launch
- Synchronization

Tests: `basic_test`, `vecadd_test`

### Phase 1B: Algorithm Validation (✅ Complete)
- Linear algebra operations
- Neural network primitives
- Memory access patterns

Tests: `sgemm_test`, `dotproduct_test`, `relu_test`, `conv3_test`

### Phase 1C: Advanced Features (✅ Complete)
- Shared memory
- Memory fences
- Thread cooperation
- Control flow divergence

Tests: `sgemm2_test`, `fence_test`, `cta_test`, `diverge_test`

### Phase 1D: Stress Testing (✅ Complete)
- Computational stress
- Memory stress
- Comprehensive demo

Tests: `madmax_test`, `mstress_test`, `demo_test`

## Running Tests

```bash
# Set environment
export VORTEX_HOME=/path/to/vortex
export LD_LIBRARY_PATH=$VORTEX_HOME/build/runtime:runtime/build
export VORTEX_DRIVER=simx

# Build and run a test
cd tests/vecadd_test
make clean && make
./run.sh

# Or use the test script
./run.sh
```

## Test Results

All Phase 1 tests are passing, verifying that:
- ✅ HIP runtime API correctly maps to Vortex API
- ✅ Metadata extraction from DWARF works
- ✅ Argument marshaling handles all data types
- ✅ Kernel registration and lazy loading works
- ✅ Memory operations are correct
- ✅ Grid/block execution model works
- ✅ Shared memory functions correctly
- ✅ Control flow divergence is handled

## What's Next: Phase 2

**Phase 2 (Compiler Integration)** will add the missing piece:
- LLVM pass or Clang plugin
- Transform HIP `__global__` kernels → Vortex format
- Automatic metadata generation from HIP source
- Handle HIP-specific constructs (`threadIdx`, `blockIdx`, `__shared__`, etc.)

**Phase 2 tests** will use actual HIP kernels and validate the full compilation pipeline.

## Summary

Phase 1 successfully validates:
1. ✅ HIP runtime library implementation
2. ✅ API mapping to Vortex
3. ✅ Metadata-driven marshaling
4. ✅ End-to-end execution on Vortex simulator

**Phase 1 is complete and provides a solid foundation for Phase 2 compiler work.**
