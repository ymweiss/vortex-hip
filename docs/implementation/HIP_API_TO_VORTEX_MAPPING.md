# HIP API to Vortex Mapping Strategy

**Status:** Implementation Ready
**Date:** 2025-11-15

---

## Overview

This document describes how HIP API calls and kernel syntax are transformed to Vortex runtime calls through a multi-stage compilation pipeline.

## Key Insight: Two Separate Transformation Paths

HIP code has two distinct components that are handled differently:

1. **Host API calls** (`hipMalloc`, `hipMemcpy`, etc.) → Handled by **header files**
2. **Kernel syntax** (`<<<>>>`, `threadIdx`, etc.) → Handled by **Polygeist + our MLIR pass**

---

## Path 1: Host API Calls (Header-Based Mapping)

### Implementation: `runtime/include/hip/hip_runtime.h`

The HIP runtime API is implemented as **inline functions** that directly call Vortex API. This follows the standard HIP approach used by ROCm and CUDA backends.

### Example Transformations

#### hipMalloc
```cpp
// HIP source code
float* d_ptr;
hipMalloc(&d_ptr, 1024);

// After C preprocessor (our hip_runtime.h inlines this)
float* d_ptr;
vx_mem_alloc(__hip_vortex_device, 1024, (uint64_t*)&d_ptr);

// After Polygeist (just a regular function call)
func.call @vx_mem_alloc(%device, %c1024, %ptr) : (...) -> i32
```

#### hipMemcpy
```cpp
// HIP source code
hipMemcpy(d_ptr, h_ptr, 1024, hipMemcpyHostToDevice);

// After C preprocessor
vx_copy_to_dev(__hip_vortex_device, (uint64_t)d_ptr, h_ptr, 1024);

// After Polygeist
func.call @vx_copy_to_dev(%device, %d_ptr, %h_ptr, %c1024) : (...) -> i32
```

#### hipDeviceSynchronize
```cpp
// HIP source code
hipDeviceSynchronize();

// After C preprocessor
vx_ready_wait(__hip_vortex_device, -1);

// After Polygeist
func.call @vx_ready_wait(%device, %c_neg1) : (...) -> i32
```

### Complete HIP Host API Mapping

| HIP API Call | Vortex API Call | Implementation |
|--------------|-----------------|----------------|
| `hipMalloc(&ptr, size)` | `vx_mem_alloc(device, size, &ptr)` | Inline function |
| `hipFree(ptr)` | `vx_mem_free(device, ptr)` | Inline function |
| `hipMemcpy(dst, src, sz, H2D)` | `vx_copy_to_dev(device, dst, src, sz)` | Inline function |
| `hipMemcpy(dst, src, sz, D2H)` | `vx_copy_from_dev(device, dst, src, sz)` | Inline function |
| `hipDeviceSynchronize()` | `vx_ready_wait(device, -1)` | Inline function |
| `hipSetDevice(id)` | `vx_dev_open(&device)` | Inline function |

**Key Point:** Polygeist never sees HIP API calls - the C preprocessor transforms them to Vortex calls before compilation.

---

## Path 2: Kernel Syntax (Compiler-Based Transformation)

### Kernel Launch: `<<<>>>` Syntax

The kernel launch syntax is **not** handled by the preprocessor - it's parsed by Polygeist's CUDA support.

```cpp
// HIP source code
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx;
}

int main() {
    kernel<<<256, 256>>>(d_data);
}
```

### Stage 1: Polygeist (--cuda-lower flag)

Polygeist recognizes `<<<>>>` syntax through its built-in CUDA support:

```mlir
// After Polygeist with --cuda-lower
gpu.launch_func @kernel
    blocks in (%c256, %c1, %c1)
    threads in (%c256, %c1, %c1)
    args(%d_data : memref<?xf32>)
```

**Device code is converted to GPU dialect:**
```mlir
gpu.func @kernel(%arg0: memref<?xf32>) kernel {
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %bdim = gpu.block_dim x

    %bid_times_bdim = arith.muli %bid, %bdim : index
    %idx = arith.addi %tid, %bid_times_bdim : index

    // ... kernel body ...
}
```

### Stage 2: Our GPUToVortexLLVM Pass

Our custom MLIR pass converts GPU dialect operations to Vortex runtime calls:

#### Host-Side: Kernel Launch
```mlir
// Input: GPU Dialect
gpu.launch_func @kernel
    blocks in (%c256, %c1, %c1)
    threads in (%c256, %c1, %c1)
    args(%d_data : memref<?xf32>)

// Output: LLVM Dialect with Vortex calls
llvm.call @vx_upload_kernel_bytes(%device, %kernel_binary, %size)
llvm.call @vx_copy_to_dev(%device, %args_dev_addr, %args_struct, %args_size)
llvm.call @vx_start(%device)
llvm.call @vx_ready_wait(%device, %timeout)
```

#### Device-Side: Thread Indexing
```mlir
// Input: GPU Dialect
%tid = gpu.thread_id x
%bid = gpu.block_id x

// Output: LLVM Dialect with Vortex intrinsics
%tid = llvm.call @vx_thread_id() : () -> i32
%warp_id = llvm.call @vx_warp_id() : () -> i32
%num_threads = llvm.call @vx_num_threads() : () -> i32
// Compute block_id from warp_id and thread counts...
```

#### Device-Side: Synchronization
```mlir
// Input: GPU Dialect
gpu.barrier

// Output: LLVM Dialect with Vortex barrier
%bar_id = llvm.mlir.constant(0 : i32) : i32
%num_threads = llvm.mlir.constant(256 : i32) : i32
llvm.call @vx_barrier(%bar_id, %num_threads) : (i32, i32) -> ()
```

### Complete Kernel Syntax Mapping

| HIP/CUDA Construct | GPU Dialect (Polygeist) | Vortex API (Our Pass) |
|-------------------|-------------------------|----------------------|
| `kernel<<<g,b>>>()` | `gpu.launch_func` | `vx_upload_kernel_bytes()` + `vx_start()` + `vx_ready_wait()` |
| `threadIdx.x` | `gpu.thread_id x` | `vx_thread_id()` |
| `blockIdx.x` | `gpu.block_id x` | Computed from `vx_warp_id()` |
| `blockDim.x` | `gpu.block_dim x` | Constant or passed as argument |
| `__syncthreads()` | `gpu.barrier` | `vx_barrier(bar_id, num_threads)` |

**Key Point:** Polygeist handles the `<<<>>>` syntax transformation. Our pass handles the GPU→Vortex mapping.

---

## Complete Compilation Pipeline

```bash
# Input: HIP source file
# user_code.hip

# Step 1: C Preprocessing (automatic)
# - Includes hip/hip_runtime.h
# - Inlines hipMalloc → vx_mem_alloc, etc.
# - Preserves <<<>>> syntax

# Step 2: Polygeist compilation
cgeist user_code.hip \
    -I runtime/include \              # Find our hip_runtime.h
    --cuda-lower \                    # Convert <<<>>> to gpu.launch_func
    -resource-dir $(clang -print-resource-dir) \
    -S -o user_code.mlir

# Result: MLIR with:
#   func.call @vx_mem_alloc (from hipMalloc)
#   gpu.launch_func (from <<<>>>)
#   gpu.thread_id (from threadIdx in kernel)

# Step 3: Standard MLIR passes (if needed)
mlir-opt user_code.mlir \
    --convert-affine-for-to-gpu \     # Parallelize loops (if any)
    -o user_code_gpu.mlir

# Step 4: Our custom pass
mlir-opt user_code_gpu.mlir \
    --convert-gpu-to-vortex-llvm \    # GPU dialect → Vortex calls
    -o user_code_llvm.mlir

# Result: LLVM Dialect with only vx_* calls

# Step 5: MLIR to LLVM IR
mlir-translate user_code_llvm.mlir \
    --mlir-to-llvmir \
    -o user_code.ll

# Step 6: RISC-V compilation
llvm-vortex/bin/clang user_code.ll \
    -target riscv32 \
    -o kernel.vxbin
```

---

## Developer Responsibilities

### Runtime Header (Week 1, Monday - 4 hours)

**File:** `runtime/include/hip/hip_runtime.h`

**Implement:**
- Error types and constants
- Inline functions for HIP API → Vortex API mapping
- Device attributes (`__global__`, `__device__`, `__host__`)
- dim3 type for kernel launch configuration

**Completed:** ✅ Template created (see runtime/include/hip/hip_runtime.h)

### Developer A: Thread Model & Kernel Launch (Weeks 2-3)

**Implement MLIR pass pattern matching:**
- `gpu.thread_id` → `vx_thread_id()`
- `gpu.block_id` → computed from `vx_warp_id()`
- `gpu.barrier` → `vx_barrier()`
- `gpu.launch_func` → `vx_upload_kernel_bytes()` + `vx_start()` + `vx_ready_wait()`

### Developer B: Memory Operations & Argument Marshaling (Weeks 2-3)

**Implement MLIR pass pattern matching:**
- `gpu.alloc` (shared) → shared memory allocation
- Address space mapping (global, shared, local)
- Kernel argument structure creation
- Argument marshaling for `gpu.launch_func`

---

## Testing Strategy

### Test 1: Verify Header Preprocessing

```bash
# Preprocess only to see vx_* calls
clang -E hip_tests/simple_malloc_test.hip \
    -I runtime/include \
    | grep -A2 "hipMalloc"

# Should show: vx_mem_alloc call
```

### Test 2: Verify Polygeist CUDA Lowering

```bash
# Compile with Polygeist
cgeist hip_tests/simple_malloc_test.hip \
    -I runtime/include \
    --cuda-lower \
    -S -o test.mlir

# Check for GPU dialect operations
grep "gpu.launch_func" test.mlir
grep "gpu.thread_id" test.mlir
```

### Test 3: Verify Our Pass Works

```bash
# Apply our custom pass
mlir-opt test.mlir \
    --convert-gpu-to-vortex-llvm \
    -o test_vortex.mlir

# Check for Vortex calls
grep "vx_upload_kernel_bytes" test_vortex.mlir
grep "vx_thread_id" test_vortex.mlir
grep "vx_barrier" test_vortex.mlir
```

---

## Why This Design Works

### 1. Separation of Concerns
- **Preprocessor** handles HIP API (simple text substitution)
- **Polygeist** handles kernel syntax (built-in CUDA support)
- **Our pass** handles Vortex-specific mappings (custom logic)

### 2. Leverages Existing Tools
- Standard C preprocessor (no custom parsing)
- Polygeist's mature CUDA support (no need to reimplement)
- Standard MLIR infrastructure (dialect conversion framework)

### 3. Maintainability
- HIP API changes → update header file
- Vortex API changes → update our MLIR pass
- Clear boundaries between components

### 4. Follows HIP Standard
- This is exactly how ROCm and CUDA backends work
- Headers provide backend-specific implementations
- Compatible with existing HIP code patterns

---

## Example: Complete Transformation

### Original HIP Code
```cpp
#include <hip/hip_runtime.h>

__global__ void add(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, 1024);
    hipMalloc(&d_b, 1024);
    hipMalloc(&d_c, 1024);

    add<<<4, 256>>>(d_a, d_b, d_c, 1024);

    hipDeviceSynchronize();
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
```

### After Preprocessing
```cpp
// hip_runtime.h has been inlined

int main() {
    float *d_a, *d_b, *d_c;
    vx_mem_alloc(__hip_vortex_device, 1024, (uint64_t*)&d_a);
    vx_mem_alloc(__hip_vortex_device, 1024, (uint64_t*)&d_b);
    vx_mem_alloc(__hip_vortex_device, 1024, (uint64_t*)&d_c);

    add<<<4, 256>>>(d_a, d_b, d_c, 1024);  // Still has <<<>>>

    vx_ready_wait(__hip_vortex_device, -1);
    vx_mem_free(__hip_vortex_device, (uint64_t)d_a);
    vx_mem_free(__hip_vortex_device, (uint64_t)d_b);
    vx_mem_free(__hip_vortex_device, (uint64_t)d_c);
}
```

### After Polygeist
```mlir
// Host code - regular function calls
func.func @main() {
    %d_a = ... allocate pointer ...
    func.call @vx_mem_alloc(%device, %c1024, %d_a)
    func.call @vx_mem_alloc(%device, %c1024, %d_b)
    func.call @vx_mem_alloc(%device, %c1024, %d_c)

    // Kernel launch - GPU dialect
    gpu.launch_func @add
        blocks in (%c4, %c1, %c1)
        threads in (%c256, %c1, %c1)
        args(%d_a, %d_b, %d_c, %c1024 : ...)

    func.call @vx_ready_wait(%device, %c_neg1)
    func.call @vx_mem_free(%device, %d_a)
    func.call @vx_mem_free(%device, %d_b)
    func.call @vx_mem_free(%device, %d_c)
}

// Device code - GPU dialect
gpu.func @add(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
              %arg2: memref<?xf32>, %arg3: i32) kernel {
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %bdim = gpu.block_dim x
    // ... kernel logic ...
}
```

### After Our GPUToVortexLLVM Pass
```mlir
// All function calls are now to Vortex API
llvm.func @main() {
    llvm.call @vx_mem_alloc(...)
    llvm.call @vx_mem_alloc(...)
    llvm.call @vx_mem_alloc(...)

    // Kernel launch expanded to multiple Vortex calls
    llvm.call @vx_upload_kernel_bytes(%device, %kernel_bin, %size)
    llvm.call @vx_copy_to_dev(%device, %args_addr, %args_struct, %args_size)
    llvm.call @vx_start(%device)
    llvm.call @vx_ready_wait(%device, %timeout)

    llvm.call @vx_mem_free(...)
    llvm.call @vx_mem_free(...)
    llvm.call @vx_mem_free(...)
}

// Device code with Vortex intrinsics
llvm.func @add(...) {
    %tid = llvm.call @vx_thread_id() : () -> i32
    %warp_id = llvm.call @vx_warp_id() : () -> i32
    // Compute block_id from warp_id...
    // ... kernel logic ...
}
```

---

## References

- **HIP Programming Guide**: https://rocm.docs.amd.com/projects/HIP/
- **Polygeist Documentation**: https://polygeist.llvm.org/
- **MLIR GPU Dialect**: https://mlir.llvm.org/docs/Dialects/GPU/
- **Vortex Runtime API**: `vortex/runtime/include/vortex.h`

---

**Status:** Design Complete, Header Template Implemented
**Next Steps:** Week 1 Tuesday - Test HIP syntax with Polygeist
