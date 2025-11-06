# chipStar Architecture: Complete Technical Analysis

## Executive Summary

**chipStar** is a HIP (Heterogeneous-computing Interface for Portability) implementation that enables compiling and running HIP and CUDA applications on platforms supporting SPIR-V as the intermediate representation. Unlike GPU-specific implementations, chipStar uses:

- **Clang's built-in HIP support** (no custom frontend modifications)
- **LLVM IR → SPIR-V translation** via SPIRV-LLVM-Translator
- **OpenCL or Level Zero** as backend runtime
- **Custom LLVM passes** to bridge HIP/CUDA semantics to OpenCL/SPIR-V

**Key Insight**: chipStar doesn't modify Clang. It leverages Clang's native HIP support (since version 14) and applies transformations at the LLVM IR level before converting to SPIR-V.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Compilation Pipeline](#compilation-pipeline)
3. [HIP to SPIR-V Translation](#hip-to-spir-v-translation)
4. [Clang Frontend Integration](#clang-frontend-integration)
5. [Kernel Launch Syntax Handling](#kernel-launch-syntax-handling)
6. [Built-in Functions Implementation](#built-in-functions-implementation)
7. [LLVM Passes](#llvm-passes)
8. [Runtime Architecture](#runtime-architecture)
9. [Complete Example Walkthrough](#complete-example-walkthrough)

---

## Architecture Overview

### Component Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    HIP/CUDA Application                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Clang with Native HIP Support                   │
│  (Handles <<<>>> syntax, __device__, __global__, etc.)      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓ (Device Code Path)
┌─────────────────────────────────────────────────────────────┐
│                  LLVM IR (SPIR-V Triple)                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            Link chipStar Bitcode Library                     │
│   (HIP device functions → OpenCL functions)                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│           chipStar Custom LLVM Passes                        │
│  (Transform HIP/CUDA constructs to SPIR-V compatible)       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            SPIRV-LLVM-Translator                             │
│              (LLVM IR → SPIR-V)                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              SPIR-V Binary (in Fat Binary)                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓ (Embedded in host binary)
┌─────────────────────────────────────────────────────────────┐
│             chipStar Runtime (libCHIP.so)                    │
│                OpenCL or Level Zero Backend                   │
└─────────────────────────────────────────────────────────────┘
```

### Two-Part System

1. **Compilation** (compile-time)
   - Uses standard Clang with HIP support
   - No custom compiler modifications
   - Applies LLVM transformation passes
   - Produces SPIR-V embedded in host binary

2. **Runtime** (execution-time)
   - Implements HIP API functions
   - Loads and JIT-compiles SPIR-V
   - Manages OpenCL/Level Zero devices
   - Handles memory, streams, events

---

## Compilation Pipeline

### Complete 7-Step Process

From the chipStar documentation (`docs/Development.md:10-17`), the compilation pipeline is:

```bash
# 1. Device-mode preprocessing
clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
    -o kernel-hip-spirv64-generic.cui -x hip kernel.hip

# 2. Device-mode compilation to LLVM IR
clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
    -o kernel-hip-spirv64-generic.bc -x hip-cpp-output kernel-hip-spirv64-generic.cui

# 3. Link chipStar bitcode library
clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
    -mlink-builtin-bitcode /path/lib/hip-device-lib/hipspv-spirv64.bc \
    -o kernel-hip-spirv64-generic.bc -x ir kernel-hip-spirv64-generic.bc

# 4. Link multiple translation units (if any)
llvm-link kernel-hip-spirv64-generic.bc -o kernel-hip-spirv64-generic-link.bc

# 5. Run chipStar LLVM transformation passes
opt kernel-hip-spirv64-generic-link.bc \
    -load-pass-plugin /path/lib/libLLVMHipSpvPasses.so \
    -passes=hip-post-link-passes \
    -o kernel-hip-spirv64-generic-lower.bc

# 6. Convert LLVM IR to SPIR-V
llvm-spirv --spirv-max-version=1.1 --spirv-ext=+all \
    kernel-hip-spirv64-generic-lower.bc \
    -o kernel-hip-spirv64-generic.out

# 7. Bundle SPIR-V into fat binary
clang-offload-bundler -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux,hip-spirv64----generic \
    -inputs=/dev/null,kernel-hip-spirv64-generic.out \
    -outputs=kernel.hipfb

# Host-mode compilation (separate path)
# 8. Compile host code with embedded fat binary
clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64 \
    -fcuda-include-gpubinary kernel.hipfb \
    -o kernel-host.bc -x hip kernel.hip

# 9. Assemble host binary
clang -cc1as -triple x86_64-unknown-linux-gnu -filetype obj \
    -o kernel.o kernel-host.s
```

### Key Observations

1. **Dual Compilation**: Device code and host code compiled separately
2. **SPIR-V Target**: Uses `spirv64` triple (recognized by Clang)
3. **Bitcode Library**: HIP device functions implemented in OpenCL C
4. **Fat Binary Format**: Clang's offload bundle (can contain multiple targets)
5. **Magic Global**: SPIR-V embedded as global variable in host binary

---

## HIP to SPIR-V Translation

### The Translation Challenge

**Problem**: HIP (based on CUDA) has constructs incompatible with SPIR-V/OpenCL:

| HIP/CUDA Construct | SPIR-V/OpenCL Issue |
|--------------------|---------------------|
| `extern __shared__ T x[]` | Dynamic local memory must be kernel arg |
| `__syncthreads()` | Different barrier semantics |
| `printf()` | Different printf mechanism |
| Global scope variables | Requires special handling |
| Switch on non-standard int widths | SPIR-V doesn't support i4, i12, etc. |
| Zero-length arrays | Invalid in SPIR-V |
| Warp intrinsics | No direct mapping to OpenCL |
| Texture objects | Different image/sampler model |

### chipStar Bitcode Library

**Location**: `bitcode/devicelib.cl` (55,889 lines of OpenCL C)

**Purpose**: Implement HIP device-side functions using OpenCL equivalents

**Example Implementations**:

```c
// From bitcode/devicelib.cl:74-92

// Built-in variables are mapped to OpenCL functions
extern "C" __device__ size_t _Z12get_local_idj(uint);
__DEVICE__ uint __hip_get_thread_idx_x() { return _Z12get_local_idj(0); }
__DEVICE__ uint __hip_get_thread_idx_y() { return _Z12get_local_idj(1); }
__DEVICE__ uint __hip_get_thread_idx_z() { return _Z12get_local_idj(2); }

extern "C" __device__ size_t _Z12get_group_idj(uint);
__DEVICE__ uint __hip_get_block_idx_x() { return _Z12get_group_idj(0); }
__DEVICE__ uint __hip_get_block_idx_y() { return _Z12get_group_idj(1); }
__DEVICE__ uint __hip_get_block_idx_z() { return _Z12get_group_idj(2); }

extern "C" __device__ size_t _Z14get_local_sizej(uint);
__DEVICE__ uint __hip_get_block_dim_x() { return _Z14get_local_sizej(0); }
__DEVICE__ uint __hip_get_block_dim_y() { return _Z14get_local_sizej(1); }
__DEVICE__ uint __hip_get_block_dim_z() { return _Z14get_local_sizej(2); }

extern "C" __device__ size_t _Z14get_num_groupsj(uint);
__DEVICE__ uint __hip_get_grid_dim_x() { return _Z14get_num_groupsj(0); }
__DEVICE__ uint __hip_get_grid_dim_y() { return _Z14get_num_groupsj(1); }
__DEVICE__ uint __hip_get_grid_dim_z() { return _Z14get_num_groupsj(2); }
```

**Key Functions Provided**:

- **Math functions**: Using OCML (ROCm OpenCL Math Library)
- **Atomic operations**: Native + emulated variants
- **Warp primitives**: Using OpenCL subgroup extensions
- **Bit manipulation**: `__brev`, `__ffs`, `__popc`, etc.
- **Special functions**: `__syncthreads()` → `barrier(CLK_LOCAL_MEM_FENCE)`

---

## Clang Frontend Integration

### Clang's Native HIP Support

**Critical Point**: chipStar does NOT modify Clang. It uses Clang's built-in HIP support.

**Since Clang 14**:
- Native understanding of HIP syntax
- Recognizes `__global__`, `__device__`, `__host__`, `__shared__`
- Handles `<<<>>>` kernel launch syntax
- Supports SPIR-V target triple (`spirv64`)
- Manages dual compilation (host + device)

**From Clang's perspective**:

```cpp
// This is standard HIP code that Clang natively understands
__global__ void kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] *= 2.0f;
}

int main() {
    float* d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));

    // Clang recognizes this syntax
    kernel<<<10, 256>>>(d_data);

    hipFree(d_data);
}
```

### No Custom Frontend Modifications

chipStar provides:

1. **Header files** (`include/hip/*.h`)
   - Define HIP API functions
   - Map built-in variables to OpenCL equivalents
   - Provide type definitions

2. **Compiler wrapper** (`bin/cucc.py`)
   - Translates nvcc arguments to hipcc
   - Useful for CUDA compatibility

3. **CMake integration**
   - Configures correct compiler flags
   - Links runtime library

**But it does NOT**:
- ❌ Patch Clang source code
- ❌ Add new parser rules
- ❌ Modify lexer/semantic analysis
- ❌ Change code generation

---

## Kernel Launch Syntax Handling

### The `<<<>>>` Syntax

**Question**: How does `kernel<<<gridDim, blockDim, sharedMem, stream>>>(args)` work?

**Answer**: Clang's native HIP support transforms it during parsing.

### Clang's Transformation

Clang converts:

```cpp
kernel<<<dim3(10, 1, 1), dim3(256, 1, 1), 0, 0>>>(arg1, arg2);
```

Into:

```cpp
__hipPushCallConfiguration(dim3(10, 1, 1), dim3(256, 1, 1), 0, 0);
kernel(arg1, arg2);
__hipPopCallConfiguration();
```

### chipStar Runtime Implementation

**Location**: `src/CHIPBindings.cc`

```cpp
// From src/CHIPBindings.cc (conceptual - actual implementation varies)

hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                       size_t sharedMem, hipStream_t stream) {
    // Store launch configuration in thread-local storage
    ConfigStack.push({gridDim, blockDim, sharedMem, stream});
    return hipSuccess;
}

hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                      size_t* sharedMem, hipStream_t* stream) {
    auto config = ConfigStack.pop();
    *gridDim = config.gridDim;
    *blockDim = config.blockDim;
    *sharedMem = config.sharedMem;
    *stream = config.stream;
    return hipSuccess;
}
```

### Alternative: `hipLaunchKernelGGL` Macro

chipStar also supports explicit launch API:

```cpp
// From include/hip/spirv_hip.hh:61-69

#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads,          \
                                   memPerBlock, streamId, ...)                 \
  do {                                                                         \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(      \
        __VA_ARGS__);                                                          \
  } while (0)

#define hipLaunchKernelGGL(kernelName, ...)                                    \
  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)
```

**Usage**:

```cpp
// Instead of:
kernel<<<gridDim, blockDim>>>(args);

// Can use:
hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, 0, args);
```

---

## Built-in Functions Implementation

### Property-Based Accessors

**Location**: `include/hip/spirv_hip.hh:94-130`

**Mechanism**: Uses C++ `__declspec(property)` with getter functions

```cpp
// From include/hip/spirv_hip.hh:94-102

#define __HIP_DEVICE_BUILTIN(DIMENSION, FUNCTION)                              \
  __declspec(property(get = __get_##DIMENSION)) uint DIMENSION;                \
  __DEVICE__ uint __get_##DIMENSION(void) { return FUNCTION; }

struct __hip_builtin_threadIdx_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_thread_idx_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_thread_idx_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_thread_idx_z());
};
```

**Expands to**:

```cpp
struct __hip_builtin_threadIdx_t {
  __declspec(property(get = __get_x)) uint x;
  __DEVICE__ uint __get_x(void) { return __hip_get_thread_idx_x(); }

  __declspec(property(get = __get_y)) uint y;
  __DEVICE__ uint __get_y(void) { return __hip_get_thread_idx_y(); }

  __declspec(property(get = __get_z)) uint z;
  __DEVICE__ uint __get_z(void) { return __hip_get_thread_idx_z(); }
};
```

### Global Variable Declaration

```cpp
// From include/hip/spirv_hip.hh:126-130

extern const __device__ __attribute__((weak))
__hip_builtin_threadIdx_t threadIdx;

extern const __device__ __attribute__((weak))
__hip_builtin_blockIdx_t blockIdx;

extern const __device__ __attribute__((weak))
__hip_builtin_blockDim_t blockDim;

extern const __device__ __attribute__((weak))
__hip_builtin_gridDim_t gridDim;
```

### Mapping to OpenCL Built-ins

**The Crucial Link**:

```cpp
// From include/hip/spirv_hip.hh:74-92

// OpenCL function prototypes (mangled names)
extern "C" __device__ size_t _Z12get_local_idj(uint);
extern "C" __device__ size_t _Z12get_group_idj(uint);
extern "C" __device__ size_t _Z14get_local_sizej(uint);
extern "C" __device__ size_t _Z14get_num_groupsj(uint);

// Wrapper functions
__DEVICE__ uint __hip_get_thread_idx_x() { return _Z12get_local_idj(0); }
__DEVICE__ uint __hip_get_block_idx_x() { return _Z12get_group_idj(0); }
__DEVICE__ uint __hip_get_block_dim_x() { return _Z14get_local_sizej(0); }
__DEVICE__ uint __hip_get_grid_dim_x() { return _Z14get_num_groupsj(0); }
```

### Complete Translation Chain

```
User Code:
    int tid = threadIdx.x;

Property Getter:
    int tid = threadIdx.__get_x();

Inline Function:
    int tid = __hip_get_thread_idx_x();

OpenCL Call:
    int tid = _Z12get_local_idj(0);

LLVM IR:
    %tid = call i64 @_Z12get_local_idj(i32 0)

SPIR-V:
    OpExtInst %uint %opencl_std get_local_id %uint_0
```

### Function Name Mangling

**OpenCL C++ name mangling** (Itanium ABI):

```
get_local_id(unsigned int)  → _Z12get_local_idj
get_group_id(unsigned int)  → _Z12get_group_idj
get_local_size(unsigned int) → _Z14get_local_sizej
get_num_groups(unsigned int) → _Z14get_num_groupsj
```

**Breakdown**:
- `_Z` = mangling prefix
- `12` = length of function name ("get_local_id" = 12 chars)
- `get_local_id` = function name
- `j` = unsigned int type code

---

## LLVM Passes

### Complete Pass List

**Location**: `llvm_passes/HipPasses.cpp:119-195`

chipStar applies **18 custom LLVM transformation passes** plus standard optimization passes.

### Pass Pipeline Order

```cpp
// From llvm_passes/HipPasses.cpp:119-195

static void addFullLinkTimePasses(ModulePassManager &MPM) {
  MPM.addPass(HipFixOpenCLMDPass());                    // #1
  MPM.addPass(HipSanityChecksPass());                   // #2
  MPM.addPass(HipEmitLoweredNamesPass());               // #3
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());        // #4
  MPM.addPass(HipLowerSwitchPass());                    // #5
  MPM.addPass(HipDynMemExternReplaceNewPass());         // #6
  MPM.addPass(HipLowerZeroLengthArraysPass());          // #7
  MPM.addPass(RemoveNoInlineOptNoneAttrsPass());        // #8 (again)
  MPM.addPass(ModuleInlinerWrapperPass());              // #9
  MPM.addPass(SROAPass());                              // #10
  MPM.addPass(HipTextureLoweringPass());                // #11
  MPM.addPass(HipPrintfToOpenCLPrintfPass());           // #12
  MPM.addPass(HipDefrostPass());                        // #13
  MPM.addPass(HipLowerMemsetPass());                    // #14
  MPM.addPass(HipAbortPass());                          // #15
  MPM.addPass(HipGlobalVariablesPass());                // #16
  MPM.addPass(HipWarpsPass());                          // #17
  MPM.addPass(HipKernelArgSpillerPass());               // #18
  MPM.addPass(HipStripUsedIntrinsicsPass());            // #19
  MPM.addPass(InternalizePass());                       // #20
  MPM.addPass(DCEPass());                               // #21
  MPM.addPass(GlobalDCEPass());                         // #22
  MPM.addPass(InferAddressSpacesPass());                // #23
  MPM.addPass(HipIGBADetectorPass());                   // #24
  MPM.addPass(HipPromoteIntsPass());                    // #25
}
```

### Detailed Pass Descriptions

#### 1. HipFixOpenCLMDPass
**Purpose**: Fix OpenCL version metadata mismatch
**Issue**: Clang inserts `opencl.ocl.version` 0.0 in HIP mode, bitcode library has 2.0
**Solution**: Force OpenCL version to 2.0
**Location**: `llvm_passes/HipPasses.cpp:84-103`

#### 2. HipSanityChecksPass
**Purpose**: Validate LLVM IR before transformation
**Checks**: Function signatures, kernel attributes, type correctness
**Location**: `llvm_passes/HipSanityChecks.cpp`

#### 3. HipEmitLoweredNamesPass
**Purpose**: Generate lowered names for hipRTC
**Function**: Required for `hiprtcGetLoweredName()` API
**Location**: `llvm_passes/HipEmitLoweredNames.cpp`

#### 4. RemoveNoInlineOptNoneAttrsPass
**Purpose**: Remove `noinline` and `optnone` attributes
**Reason**: Enable optimization of device code
**Location**: `llvm_passes/HipPasses.cpp:62-73`

#### 5. HipLowerSwitchPass
**Purpose**: Lower switch instructions with non-standard int widths
**Issue**: SPIR-V doesn't support i4, i12, etc.
**Solution**: Convert to supported widths (i8, i16, i32, i64)
**Location**: `llvm_passes/HipLowerSwitch.cpp`

**Example**:
```llvm
; Before:
switch i4 %val, label %default [...]

; After:
%extended = zext i4 %val to i8
switch i8 %extended, label %default [...]
```

#### 6. HipDynMemExternReplaceNewPass
**Purpose**: Replace dynamic shared memory with kernel argument
**Issue**: `extern __shared__ T x[]` not supported in OpenCL
**Solution**: Convert to kernel parameter
**Location**: `llvm_passes/HipDynMem.cpp`

**Transformation**:
```cpp
// Before:
__global__ void kernel() {
    extern __shared__ float shmem[];
    shmem[threadIdx.x] = ...;
}

// After (LLVM IR):
define spir_kernel void @kernel(
    ...,
    i8 addrspace(3)* %__chip_dyn_lds  // Added parameter
) {
    %shmem = bitcast i8 addrspace(3)* %__chip_dyn_lds to float addrspace(3)*
    ...
}
```

#### 7. HipLowerZeroLengthArraysPass
**Purpose**: Remove zero-length array types
**Issue**: SPIR-V doesn't support `[0 x T]`
**Solution**: Replace with valid type
**Location**: `llvm_passes/HipLowerZeroLengthArrays.cpp`

#### 8-10. Optimization Passes
**RemoveNoInlineOptNoneAttrs** (again): Before inlining
**ModuleInliner**: Aggressive inlining (threshold 1000)
**SROA**: Scalar Replacement of Aggregates

**Purpose**: Prepare for texture lowering (which requires inlined code)

#### 11. HipTextureLoweringPass
**Purpose**: Convert `hipTextureObject_t` to OpenCL image+sampler
**Complexity**: Transforms function signatures
**Location**: `llvm_passes/HipTextureLowering.cpp`

**Example**:
```cpp
// Before:
__global__ void kernel(hipTextureObject_t tex) {
    float val = tex2D<float>(tex, x, y);
}

// After (conceptual):
__global__ void kernel(__read_only image2d_t img, sampler_t samp) {
    float val = read_imagef(img, samp, (float2)(x, y)).x;
}
```

#### 12. HipPrintfToOpenCLPrintfPass
**Purpose**: Convert CUDA/HIP printf to OpenCL printf
**Differences**: Format string handling, argument marshalling
**Location**: `llvm_passes/HipPrintf.cpp`

#### 13. HipDefrostPass
**Purpose**: Remove `freeze` instructions
**Reason**: Workaround for llvm-spirv translator
**Location**: `llvm_passes/HipDefrost.cpp`

#### 14. HipLowerMemsetPass
**Purpose**: Lower `memset` to SPIR-V compatible form
**Location**: `llvm_passes/HipLowerMemset.cpp`

#### 15. HipAbortPass
**Purpose**: Handle `abort()` calls from device code
**Mechanism**: Set global flag, check on host
**Location**: `llvm_passes/HipAbort.cpp`

**Implementation**:
```cpp
// Device code:
__device__ int32_t __chipspv_abort_called;

void abort() {
    __chipspv_abort(&__chipspv_abort_called);
}

// Runtime checks this flag after kernel execution
```

#### 16. HipGlobalVariablesPass
**Purpose**: Create accessor kernels for global variables
**Issue**: SPIR-V doesn't support direct global variable access from host
**Solution**: Generate `get` and `set` kernels
**Location**: `llvm_passes/HipGlobalVariables.cpp`

**Generated Code**:
```cpp
// Original:
__device__ int globalVar;

// Generated kernels:
__global__ void __chip_get_globalVar(int* out) {
    *out = globalVar;
}

__global__ void __chip_set_globalVar(int* in) {
    globalVar = *in;
}
```

#### 17. HipWarpsPass
**Purpose**: Handle warp-sensitive kernels
**Actions**: Annotate kernels requiring subgroup size
**Extension**: Uses `cl_intel_required_subgroup_size`
**Location**: `llvm_passes/HipWarps.cpp`

**Metadata Addition**:
```llvm
!kernel_metadata = !{
  !{!"reqd_work_group_size", i32 0, i32 0, i32 0},
  !{!"intel_reqd_sub_group_size", i32 32}  ; Added by pass
}
```

#### 18. HipKernelArgSpillerPass
**Purpose**: Reduce large kernel parameter lists
**Method**: Spill arguments to device buffer
**Threshold**: Configurable (typically >256 bytes)
**Location**: `llvm_passes/HipKernelArgSpiller.cpp`

**Transformation**:
```cpp
// Before:
__global__ void kernel(float a1, float a2, ..., float a100);

// After:
__global__ void kernel(void* arg_buffer) {
    float* args = (float*)arg_buffer;
    float a1 = args[0];
    float a2 = args[1];
    ...
}
```

#### 19. HipStripUsedIntrinsicsPass
**Purpose**: Remove `llvm.used` and `llvm.compiler.used` globals
**Reason**: Prevent DCE from removing symbols we want deleted
**Location**: `llvm_passes/HipStripUsedIntrinsics.cpp`

#### 20. InternalizePass
**Purpose**: Internalize all non-kernel functions
**Effect**: Enables DCE to remove unused functions
**Predicate**: Preserve only `spir_kernel` calling convention

```cpp
// From llvm_passes/HipPasses.cpp:55-59
static bool internalizeSPIRVFunctions(const GlobalValue &GV) {
  const auto *F = dyn_cast<Function>(&GV);
  // Returning true means preserve GV.
  return !(F && F->getCallingConv() == CallingConv::SPIR_FUNC);
}
```

#### 21-22. Dead Code Elimination
**DCEPass**: Function-level dead code elimination
**GlobalDCEPass**: Module-level global dead code elimination

#### 23. InferAddressSpacesPass
**Purpose**: Infer address spaces for pointers
**Target**: Generic address space (4)
**Effect**: Better optimization opportunities

#### 24. HipIGBADetectorPass
**Purpose**: Detect Indirect GPU Barrier Access
**Function**: Identify kernels with dynamic barrier requirements
**Location**: `llvm_passes/HipIGBADetector.cpp`

#### 25. HipPromoteIntsPass
**Purpose**: Fix InvalidBitWidth errors
**Issue**: SPIR-V doesn't support arbitrary bit widths
**Solution**: Promote to standard widths
**Location**: `llvm_passes/HipPromoteInts.cpp`

**Example**:
```llvm
; Before:
%result = add i13 %a, %b

; After:
%a_ext = zext i13 %a to i16
%b_ext = zext i13 %b to i16
%result_wide = add i16 %a_ext, %b_ext
%result = trunc i16 %result_wide to i13
```

### Pass Invocation

**Command Line**:
```bash
opt module.bc \
    -load-pass-plugin /path/libLLVMHipSpvPasses.so \
    -passes=hip-post-link-passes \
    -o module-transformed.bc
```

**Plugin Registration**:
```cpp
// From llvm_passes/HipPasses.cpp:203-242

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "hip-passes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hip-post-link-passes") {
                    addFullLinkTimePasses(MPM);
                    return true;
                  }
                  return false;
                });
          }};
}
```

---

## Runtime Architecture

### Runtime Components

**Main Files**:
- `src/CHIPBindings.cc` - HIP API implementation (hipMalloc, hipMemcpy, etc.)
- `src/CHIPBackend.{cc,hh}` - Abstract backend interface
- `src/backend/OpenCL/CHIPBackendOpenCL.cc` - OpenCL implementation
- `src/backend/Level0/CHIPBackendLevel0.cc` - Level Zero implementation
- `src/spirv.cc` - SPIR-V binary parser
- `src/CHIPDriver.cc` - Initialization/cleanup

### Application Lifetime

**From `docs/Development.md:123-134`**:

#### Initialization (Before main())

Clang inserts hidden calls:

```cpp
// Conceptual flow - actual code is auto-generated by Clang

// 1. Register fat binary
__hipRegisterFatBinary(fatbin_handle, fatbin_blob);
// - Parses offload bundle
// - Extracts SPIR-V binary
// - Compiles to device-specific code (OpenCL program / Level Zero module)
// - Parses function signatures

// 2. Register each kernel
__hipRegisterFunction(fatbin_handle, host_function_ptr, "kernel_name", ...);
// - Creates mapping: host pointer → device kernel
// - Stores kernel metadata (grid limits, register usage, etc.)

// 3. Register global variables
__hipRegisterVar(fatbin_handle, host_var_ptr, "var_name", size, ...);
// - Creates mapping: host pointer → device variable
// - Used for __device__ global variables
```

#### Runtime (During main())

```cpp
int main() {
    // HIP API calls are handled by CHIPBindings.cc

    float* d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));
    // → OpenCL: clCreateBuffer() or LevelZero: zeMemAllocDevice()

    hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);
    // → OpenCL: clEnqueueWriteBuffer() or LevelZero: zeCommandListAppendMemoryCopy()

    kernel<<<10, 256>>>(d_data);
    // → OpenCL: clSetKernelArg() + clEnqueueNDRangeKernel()
    // → LevelZero: zeKernelSetArgumentValue() + zeCommandListAppendLaunchKernel()

    hipDeviceSynchronize();
    // → OpenCL: clFinish() or LevelZero: zeCommandQueueSynchronize()

    hipFree(d_data);
    // → OpenCL: clReleaseMemObject() or LevelZero: zeMemFree()
}
```

#### Cleanup (After main())

```cpp
__hipUnregisterFatBinary(fatbin_handle);
// - Releases device programs/modules
// - Frees backend resources
// - When all modules unregistered, calls CHIPUninitialize()
```

### Backend Abstraction

```cpp
// Simplified from src/CHIPBackend.hh

class CHIPBackend {
public:
    virtual CHIPDevice* getDevice(int deviceId) = 0;
    virtual CHIPQueue* createQueue(CHIPDevice* device) = 0;
    virtual CHIPKernel* createKernel(CHIPModule* module, const std::string& name) = 0;
    virtual CHIPMemory* allocateMemory(size_t size) = 0;
    // ... many more virtual methods
};

class CHIPBackendOpenCL : public CHIPBackend {
    // Implements using OpenCL API (cl*)
};

class CHIPBackendLevel0 : public CHIPBackend {
    // Implements using Level Zero API (ze*)
};
```

### SPIR-V Parsing

**Location**: `src/spirv.cc`

**Purpose**: Extract kernel metadata from SPIR-V

**Parsed Information**:
- Kernel names
- Argument types
- Argument sizes
- Address spaces
- Image/sampler types

**Why Needed**: To properly marshal arguments when launching kernels

**Example**:
```cpp
// From src/spirv.cc (conceptual)

class SPIRVModule {
    std::map<std::string, SPIRVKernel> kernels_;

    void parse(const uint32_t* spirv_words, size_t num_words) {
        // Parse OpEntryPoint instructions
        // Parse OpTypePointer, OpTypeInt, etc.
        // Build kernel signature
    }

    SPIRVKernel* getKernel(const std::string& name) {
        return &kernels_[name];
    }
};

struct SPIRVKernel {
    std::string name;
    std::vector<SPIRVArgument> arguments;
};

struct SPIRVArgument {
    SPVTypeKind type;        // POD, Pointer, Image, Sampler
    SPVStorageClass storage; // CrossWorkgroup, Workgroup, etc.
    size_t size;
    size_t alignment;
};
```

---

## Complete Example Walkthrough

Let's trace a simple HIP program through the entire chipStar system.

### Source Code

**File**: `vector_add.hip`

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    std::cout << "Result: " << h_c[0] << " " << h_c[n-1] << std::endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```

### Compilation Steps

#### Step 1: Clang Device-Mode Compilation

```bash
clang -cc1 -triple spirv64 -aux-triple x86_64-linux-gnu \
    -D__HIP_PLATFORM_SPIRV__ \
    -I/path/chipStar/include \
    -x hip vector_add.hip \
    -emit-llvm -o vector_add-device.bc
```

**Generated LLVM IR** (simplified):

```llvm
; Device code only - host code ignored in this pass

define spir_kernel void @_Z9vectorAddPfS_S_i(
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c,
    i32 %n
) {
entry:
  %0 = call i32 @__hip_get_block_idx_x()
  %1 = call i32 @__hip_get_block_dim_x()
  %2 = mul i32 %0, %1
  %3 = call i32 @__hip_get_thread_idx_x()
  %idx = add i32 %2, %3
  %cmp = icmp slt i32 %idx, %n
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %aidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %idx
  %aval = load float, float addrspace(1)* %aidx
  %bidx = getelementptr inbounds float, float addrspace(1)* %b, i32 %idx
  %bval = load float, float addrspace(1)* %bidx
  %sum = fadd float %aval, %bval
  %cidx = getelementptr inbounds float, float addrspace(1)* %c, i32 %idx
  store float %sum, float addrspace(1)* %cidx
  br label %if.end

if.end:
  ret void
}

declare i32 @__hip_get_block_idx_x()
declare i32 @__hip_get_block_dim_x()
declare i32 @__hip_get_thread_idx_x()
```

#### Step 2: Link Bitcode Library

```bash
clang -cc1 -triple spirv64 \
    -mlink-builtin-bitcode /path/lib/hipspv-spirv64.bc \
    -x ir vector_add-device.bc \
    -o vector_add-linked.bc
```

**Effect**: Links definitions of `__hip_get_*` functions from devicelib.cl

**Now LLVM IR contains**:

```llvm
define spir_func i32 @__hip_get_thread_idx_x() {
  %call = call spir_func i64 @_Z12get_local_idj(i32 0)
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

declare spir_func i64 @_Z12get_local_idj(i32)
```

#### Step 3: Run LLVM Passes

```bash
opt vector_add-linked.bc \
    -load-pass-plugin /path/libLLVMHipSpvPasses.so \
    -passes=hip-post-link-passes \
    -o vector_add-transformed.bc
```

**Transformations Applied**:

1. **InferAddressSpaces**: Converts generic pointers to specific address spaces
2. **Inlining**: `__hip_get_thread_idx_x()` inlined into kernel
3. **DCE**: Unused functions removed

**Resulting LLVM IR** (simplified):

```llvm
define spir_kernel void @_Z9vectorAddPfS_S_i(
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c,
    i32 %n
) {
entry:
  ; Inlined __hip_get_thread_idx_x()
  %tid = call spir_func i64 @_Z12get_local_idj(i32 0)
  %tid32 = trunc i64 %tid to i32

  ; Inlined __hip_get_block_idx_x()
  %bid = call spir_func i64 @_Z12get_group_idj(i32 0)
  %bid32 = trunc i64 %bid to i32

  ; Inlined __hip_get_block_dim_x()
  %bdim = call spir_func i64 @_Z14get_local_sizej(i32 0)
  %bdim32 = trunc i64 %bdim to i32

  %blockStart = mul i32 %bid32, %bdim32
  %idx = add i32 %blockStart, %tid32
  %cmp = icmp slt i32 %idx, %n
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %aidx = getelementptr inbounds float, float addrspace(1)* %a, i32 %idx
  %aval = load float, float addrspace(1)* %aidx
  %bidx = getelementptr inbounds float, float addrspace(1)* %b, i32 %idx
  %bval = load float, float addrspace(1)* %bidx
  %sum = fadd float %aval, %bval
  %cidx = getelementptr inbounds float, float addrspace(1)* %c, i32 %idx
  store float %sum, float addrspace(1)* %cidx
  br label %if.end

if.end:
  ret void
}

; OpenCL built-in declarations
declare spir_func i64 @_Z12get_local_idj(i32)
declare spir_func i64 @_Z12get_group_idj(i32)
declare spir_func i64 @_Z14get_local_sizej(i32)
```

#### Step 4: LLVM IR to SPIR-V

```bash
llvm-spirv --spirv-max-version=1.1 --spirv-ext=+all \
    vector_add-transformed.bc \
    -o vector_add.spv
```

**Generated SPIR-V** (conceptual - actual is binary):

```spirv
OpCapability Kernel
OpCapability Addresses
OpCapability Int64
OpMemoryModel Physical64 OpenCL

; Import OpenCL.std extended instruction set
%opencl = OpExtInstImport "OpenCL.std"

; Type declarations
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%float = OpTypeFloat 32
%ptr_global_float = OpTypePointer CrossWorkgroup %float
%kernel_type = OpTypeFunction %void %ptr_global_float %ptr_global_float %ptr_global_float %uint

; Constants
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2

; Kernel entry point
OpEntryPoint Kernel %vectorAdd "vectorAdd"

; Kernel implementation
%vectorAdd = OpFunction %void None %kernel_type
%a = OpFunctionParameter %ptr_global_float
%b = OpFunctionParameter %ptr_global_float
%c = OpFunctionParameter %ptr_global_float
%n = OpFunctionParameter %uint

%entry = OpLabel

; tid = get_local_id(0)
%tid64 = OpExtInst %ulong %opencl get_local_id %uint_0
%tid = OpUConvert %uint %tid64

; bid = get_group_id(0)
%bid64 = OpExtInst %ulong %opencl get_group_id %uint_0
%bid = OpUConvert %uint %bid64

; bdim = get_local_size(0)
%bdim64 = OpExtInst %ulong %opencl get_local_size %uint_0
%bdim = OpUConvert %uint %bdim64

; idx = bid * bdim + tid
%blockStart = OpIMul %uint %bid %bdim
%idx = OpIAdd %uint %blockStart %tid

; if (idx < n)
%cmp = OpSLessThan %bool %idx %n
OpBranchConditional %cmp %if_then %if_end

%if_then = OpLabel
; a_val = a[idx]
%aidx = OpInBoundsPtrAccessChain %ptr_global_float %a %idx
%aval = OpLoad %float %aidx

; b_val = b[idx]
%bidx = OpInBoundsPtrAccessChain %ptr_global_float %b %idx
%bval = OpLoad %float %bidx

; sum = a_val + b_val
%sum = OpFAdd %float %aval %bval

; c[idx] = sum
%cidx = OpInBoundsPtrAccessChain %ptr_global_float %c %idx
OpStore %cidx %sum
OpBranch %if_end

%if_end = OpLabel
OpReturn

OpFunctionEnd
```

#### Step 5: Bundle SPIR-V

```bash
clang-offload-bundler -type=o -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux,hip-spirv64----generic \
    -inputs=/dev/null,vector_add.spv \
    -outputs=vector_add.hipfb
```

**Fat Binary Structure**:

```
┌────────────────────────────────────────┐
│         Offload Bundle Header          │
│  Magic: 0x0FF10AD                      │
│  NumBundles: 2                         │
├────────────────────────────────────────┤
│      Bundle 1: host-x86_64             │
│  (empty - placeholder)                 │
├────────────────────────────────────────┤
│    Bundle 2: hip-spirv64-generic       │
│  Size: 14,328 bytes                    │
│  Data: [SPIR-V binary blob]            │
└────────────────────────────────────────┘
```

#### Step 6: Compile Host Code

```bash
clang -cc1 -triple x86_64-unknown-linux-gnu \
    -aux-triple spirv64 \
    -fcuda-include-gpubinary vector_add.hipfb \
    -D__HIP_PLATFORM_SPIRV__ \
    -I/path/chipStar/include \
    -x hip vector_add.hip \
    -emit-llvm -o vector_add-host.bc
```

**Generated Host LLVM IR** (simplified):

```llvm
; Global variable containing the fat binary
@__hip_fatbin = internal constant [14328 x i8] c"<binary blob>", section ".hip_fatbin"

; Constructor function (runs before main)
define internal void @__hip_module_ctor() {
  %handle = call i8* @__hipRegisterFatBinary(i8* getelementptr([14328 x i8], [14328 x i8]* @__hip_fatbin, i32 0, i32 0))
  call void @__hipRegisterFunction(i8* %handle, i8* bitcast (void (...)* @_Z9vectorAddPfS_S_i to i8*), i8* getelementptr([10 x i8], [10 x i8]* c"vectorAdd\00", i32 0, i32 0), ...)
  ret void
}

; Main function
define i32 @main() {
  ; ... allocation and initialization code ...

  ; Kernel launch configuration
  call void @__hipPushCallConfiguration(<3 x i32> <i32 4, i32 1, i32 1>, <3 x i32> <i32 256, i32 1, i32 1>, i64 0, i8* null)

  ; Kernel call
  call void @_Z9vectorAddPfS_S_i.stub(float* %d_a, float* %d_b, float* %d_c, i32 1024)

  call void @__hipPopCallConfiguration(...)

  ; ... rest of main ...
}

; Stub function for kernel (calls runtime)
define void @_Z9vectorAddPfS_S_i.stub(float* %a, float* %b, float* %c, i32 %n) {
  %args = alloca [4 x i8*]
  ; Marshal arguments
  %args0 = getelementptr [4 x i8*], [4 x i8*]* %args, i32 0, i32 0
  store i8* bitcast (float** %a to i8*), i8** %args0
  ; ... store other args ...

  ; Call runtime launch function
  call i32 @hipLaunchKernel(i8* bitcast (void (...)* @_Z9vectorAddPfS_S_i to i8*), <3 x i32> ..., [4 x i8*]* %args, ...)
  ret void
}

declare void @__hipPushCallConfiguration(...)
declare void @__hipPopCallConfiguration(...)
declare i32 @hipLaunchKernel(...)
declare i8* @__hipRegisterFatBinary(i8*)
declare void @__hipRegisterFunction(...)
```

#### Step 7: Link Final Executable

```bash
clang++ vector_add-host.bc -L/path/chipStar/lib -lCHIP -o vector_add
```

### Runtime Execution

#### Initialization Phase (Before main())

```
1. Dynamic linker loads libCHIP.so

2. __hip_module_ctor() runs:

   a) __hipRegisterFatBinary(fatbin_blob)
      → CHIPBackend::registerModule(spirv_binary)

      OpenCL path:
      → clCreateProgramWithIL(spirv_binary)
      → clBuildProgram(program)

      Level Zero path:
      → zeModuleCreate(spirv_binary)

   b) Parse SPIR-V to extract kernel signatures
      → SPIRVModule::parse(spirv_binary)
      → Stores: vectorAdd(float*, float*, float*, int)

   c) __hipRegisterFunction(handle, host_ptr, "vectorAdd")
      → Maps: host function pointer → device kernel
      → Stores kernel metadata

3. main() begins execution
```

#### Kernel Launch (vectorAdd<<<4, 256>>>(...))

```
1. __hipPushCallConfiguration(dim3(4,1,1), dim3(256,1,1), 0, null)
   → Stores config in thread-local stack

2. vectorAdd.stub(d_a, d_b, d_c, 1024)
   → Marshal arguments to array
   → Call hipLaunchKernel(kernel_ptr, config, args)

3. hipLaunchKernel() in CHIPBindings.cc:

   a) __hipPopCallConfiguration()
      → Retrieve: gridDim=4, blockDim=256, sharedMem=0, stream=default

   b) Look up kernel from host_ptr
      → Find: CHIPKernel* kernel = module->getKernel("vectorAdd")

   c) Set kernel arguments:
      OpenCL:
      → clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a)
      → clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b)
      → clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c)
      → clSetKernelArg(kernel, 3, sizeof(int), &1024)

      Level Zero:
      → zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_a)
      → zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &d_b)
      → zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &d_c)
      → zeKernelSetArgumentValue(kernel, 3, sizeof(int), &1024)

   d) Enqueue kernel:
      OpenCL:
      → global_work_size = gridDim * blockDim = [1024, 1, 1]
      → local_work_size = blockDim = [256, 1, 1]
      → clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                               global_work_size, local_work_size, ...)

      Level Zero:
      → group_count = gridDim = [4, 1, 1]
      → zeKernelSetGroupSize(kernel, 256, 1, 1)
      → zeCommandListAppendLaunchKernel(cmdlist, kernel, &group_count, ...)

4. Kernel executes on device:

   OpenCL:
   - 4 work-groups, each with 256 work-items
   - Total: 1024 work-items
   - Each work-item executes vectorAdd kernel

   Inside kernel:
   - get_local_id(0) returns 0-255 (threadIdx.x)
   - get_group_id(0) returns 0-3 (blockIdx.x)
   - get_local_size(0) returns 256 (blockDim.x)
   - idx = blockIdx.x * 256 + threadIdx.x
   - c[idx] = a[idx] + b[idx]

5. Kernel completes

6. hipMemcpy copies result back to host
```

---

## Appendix: File Reference

### Key Files and Line Numbers

#### Compilation Pipeline
- `docs/Development.md:8-17` - Compilation steps
- `docs/Development.md:18-62` - Example with verbose output
- `llvm_passes/HipPasses.cpp:119-195` - Pass pipeline

#### Built-in Functions
- `include/hip/spirv_hip.hh:74-130` - threadIdx, blockIdx, etc. implementation
- `bitcode/devicelib.cl` - OpenCL device library (55,889 lines)

#### LLVM Passes
- `llvm_passes/HipAbort.cpp` - abort() handling
- `llvm_passes/HipDynMem.cpp` - Dynamic shared memory
- `llvm_passes/HipGlobalVariables.cpp` - Global variable accessors
- `llvm_passes/HipKernelArgSpiller.cpp` - Large argument lists
- `llvm_passes/HipPrintf.cpp` - printf lowering
- `llvm_passes/HipPromoteInts.cpp` - Non-standard int widths
- `llvm_passes/HipTextureLowering.cpp` - Texture objects
- `llvm_passes/HipWarps.cpp` - Warp-sensitive kernels

#### Runtime
- `src/CHIPBindings.cc` - HIP API implementation
- `src/CHIPBackend.{cc,hh}` - Backend abstraction
- `src/backend/OpenCL/CHIPBackendOpenCL.cc` - OpenCL backend
- `src/backend/Level0/CHIPBackendLevel0.cc` - Level Zero backend
- `src/spirv.cc` - SPIR-V parser
- `src/CHIPDriver.cc` - Initialization/cleanup

#### Compiler Wrapper
- `bin/cucc.py` - CUDA compatibility wrapper

---

## Comparison: chipStar vs. HIP-CPU vs. ROCm HIP

| Aspect | chipStar | HIP-CPU | ROCm HIP |
|--------|----------|---------|----------|
| **Target** | OpenCL/Level Zero devices | CPU only | AMD GPUs |
| **Compilation** | Clang → LLVM IR → SPIR-V | Macros + standard C++ | Clang → AMD GPU ISA |
| **Frontend** | Clang native HIP | No special frontend | Clang native HIP |
| **Runtime** | OpenCL/Level Zero | C++17 std::execution | AMD ROCm runtime |
| **Threading** | GPU threads | Fibers/loops | GPU threads |
| **Barriers** | GPU barriers | Fiber yield | GPU barriers |
| **Atomics** | GPU atomics | CPU atomics | GPU atomics |
| **Performance** | GPU performance | CPU performance | Optimal GPU perf |

---

## Conclusion

chipStar is a **compiler + runtime system** that:

1. **Leverages Clang's native HIP support** - No frontend modifications
2. **Uses LLVM transformation passes** - Bridges HIP/CUDA to OpenCL/SPIR-V
3. **Compiles to SPIR-V** - Portable across OpenCL and Level Zero devices
4. **Provides runtime abstraction** - Supports multiple backends
5. **Maintains compatibility** - Runs HIP and CUDA applications

**Key Innovation**: Instead of modifying the compiler frontend, chipStar applies transformations at the LLVM IR level, making it maintainable and compatible with upstream Clang releases.

**Architecture Philosophy**: Minimal invasiveness, maximum compatibility, leveraging existing tools (Clang, LLVM, SPIR-V ecosystem) rather than reinventing them.
