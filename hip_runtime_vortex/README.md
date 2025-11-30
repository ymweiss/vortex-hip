# Vortex HIP Runtime Mock

Minimal HIP runtime header for Polygeist compilation to GPU dialect MLIR.

## Why Use This Instead of Full HIP Headers?

**This mock header is intentionally used instead of the full HIP/CUDA runtime headers** to maintain precise control over what Polygeist sees during compilation. This approach:

1. **Minimizes dependencies** - Only includes what's needed for GPU dialect generation
2. **Ensures predictable output** - Full HIP headers contain thousands of declarations that can interfere with Polygeist's analysis
3. **Avoids version conflicts** - No dependency on specific HIP/CUDA SDK versions
4. **Simplifies debugging** - When something goes wrong, only ~90 lines to check vs 100,000+ lines
5. **Faster compilation** - Significantly less preprocessing work

The full AMD HIP runtime headers (`hip/include/`) exist in the repository but are **NOT used for Polygeist conversion**.

## Purpose

This header provides the minimum declarations needed for Polygeist (cgeist) to:
1. Parse HIP kernel syntax (`__global__`, `__device__`, etc.)
2. Recognize GPU built-in variables (threadIdx, blockIdx, blockDim, gridDim)
3. Support kernel launch syntax (`<<<grid, block>>>`)
4. Generate GPU dialect IR instead of treating GPU constructs as regular C++ code

## Usage

```cpp
// your_kernel.hip
#include "hip_runtime_vortex/hip_runtime.h"

__global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

void launch_kernel(float* d_data, int n, int threads) {
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(d_data, n);
}
```

Convert to GPU dialect MLIR:
```bash
./scripts/polygeist/hip-to-gpu-dialect.sh your_kernel.hip output.mlir
```

## Key Features

### 1. Clang CUDA Built-in Variables
**Critical:** Includes `__clang_cuda_builtin_vars.h` to ensure threadIdx, blockIdx, etc. are recognized as GPU operations, not regular variables.

Without this header:
```mlir
memref.global @threadIdx : memref<1x3xi32>  // WRONG - treated as global variable
```

With this header:
```mlir
%0 = gpu.thread_id x  // CORRECT - GPU dialect operation
```

### 2. HIP/CUDA Attributes
Defines `__global__`, `__device__`, `__host__`, `__shared__` as compiler attributes.

### 3. Kernel Launch Support
Provides declarations for `cudaConfigureCall` and `hipConfigureCall` required for `<<<>>>` syntax parsing.

### 4. dim3 Structure
Complete dim3 implementation with constructor for grid/block dimensions.

## What This Header Does NOT Provide

This is a **compile-time only** mock header for Polygeist. It does NOT provide:
- ❌ Actual HIP runtime implementation
- ❌ Memory management (hipMalloc, hipMemcpy, etc.)
- ❌ Device management (hipSetDevice, hipGetDeviceProperties, etc.)
- ❌ Synchronization (hipDeviceSynchronize, etc.)
- ❌ Error handling beyond type definitions

For actual HIP program compilation and execution, use the full AMD HIP runtime.

## Architecture

```
hip_runtime_vortex/
├── hip_runtime.h          # Mock HIP runtime for Polygeist
└── README.md              # This file

Purpose: Convert HIP kernel source → GPU dialect MLIR
Not for: Actual HIP program compilation/execution
```

## Comparison with Full HIP Runtime

| Feature | hip_runtime_vortex | AMD HIP Runtime |
|---------|-------------------|-----------------|
| **Kernel syntax parsing** | ✅ Yes | ✅ Yes |
| **GPU builtin variables** | ✅ Yes (via clang) | ✅ Yes |
| **Kernel launch syntax** | ✅ Yes (declarations only) | ✅ Yes (full impl) |
| **Memory management** | ❌ No | ✅ Yes |
| **Device management** | ❌ No | ✅ Yes |
| **Runtime API** | ❌ No | ✅ Yes |
| **Target** | Polygeist → MLIR | AMD GPUs / NVIDIA GPUs |
| **Size** | ~90 lines | 100,000+ lines |

## Technical Details

### Why `__clang_cuda_builtin_vars.h`?

Clang provides special handling for CUDA/HIP built-in variables through this header. It defines:
```cpp
struct __cuda_builtin_threadIdx_t {
  __device__ unsigned int __fetch_builtin_x();
  __device__ unsigned int __fetch_builtin_y();
  __device__ unsigned int __fetch_builtin_z();
  // Special member function names trigger GPU dialect generation
};
__device__ const __cuda_builtin_threadIdx_t threadIdx;
```

Polygeist recognizes these special `__fetch_builtin_*` member functions and generates GPU dialect operations.

### Polygeist .hip Extension Limitation

**Known Issue:** Polygeist doesn't properly handle `.hip` file extension.

**Workaround:** The conversion script (`hip-to-gpu-dialect.sh`) automatically converts `.hip` → `.cu` temporarily before processing.

## Related Files

- `scripts/polygeist/hip-to-gpu-dialect.sh` - Single file converter
- `scripts/polygeist/convert-all-kernels.sh` - Batch converter
- `hip_tests/kernels/` - Example kernel files
- `docs/phase2-polygeist/PHASE2A_SOLUTION.md` - Detailed technical documentation

## Example Output

Input (basic_kernel.hip):
```cpp
__global__ void basic_kernel(int* src, int* dst, int count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    dst[tid] = src[tid];
  }
}
```

Output (GPU dialect MLIR):
```mlir
gpu.module @__polygeist_gpu_module {
  gpu.func @basic_kernel(...) kernel {
    %0 = gpu.block_id x
    %1 = gpu.thread_id x
    %2 = gpu.block_dim x
    // ... rest of kernel
    gpu.return
  }
}
```

## Status

✅ **Working and Verified**
- Converts HIP kernels to GPU dialect MLIR
- Tested with basic, vecadd, and dotproduct kernels
- GPU operations correctly recognized (not memref globals)
- Ready for Phase 2B metadata extraction work
