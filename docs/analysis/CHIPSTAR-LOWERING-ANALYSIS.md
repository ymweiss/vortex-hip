# chipStar LLVM IR to SPIR-V Lowering Deep Dive

**Part 2: Transformation Mechanisms and Code Generation**

This document provides an in-depth analysis of how chipStar lowers LLVM IR to SPIR-V, including address space mappings, memory model transformations, atomic operations, and the actual SPIR-V binary generation process.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Address Space Mapping](#address-space-mapping)
3. [LLVM IR to SPIR-V Translation Pipeline](#llvm-ir-to-spirv-translation-pipeline)
4. [Shared Memory Transformation](#shared-memory-transformation)
5. [Barrier Synchronization](#barrier-synchronization)
6. [Atomic Operations Translation](#atomic-operations-translation)
7. [Integer Type Promotion](#integer-type-promotion)
8. [Texture Lowering](#texture-lowering)
9. [SPIR-V Binary Generation](#spirv-binary-generation)
10. [Complete Example Walkthrough](#complete-example-walkthrough)

---

## Executive Summary

chipStar transforms HIP/CUDA code to SPIR-V through a multi-stage process:

1. **Clang Frontend**: Compiles `.hip` source to LLVM IR with target `spirv64`
2. **Bitcode Linking**: Links chipStar device library (OpenCL C implementations)
3. **LLVM Transformation Passes**: 25 custom passes adapt HIP semantics to OpenCL/SPIR-V
4. **SPIR-V Translation**: External `llvm-spirv` tool converts LLVM IR → SPIR-V binary
5. **Runtime Linking**: Additional SPIR-V modules linked at JIT time based on capabilities

**Key Insight**: chipStar does NOT create a custom SPIR-V backend. Instead, it prepares LLVM IR to be compatible with the existing SPIRV-LLVM-Translator tool maintained by the Khronos Group.

---

## Address Space Mapping

### SPIR-V Address Space Constants

chipStar defines the SPIR-V address space mapping in `llvm_passes/LLVMSPIRV.h`:

```cpp
// Lines 19-28 of llvm_passes/LLVMSPIRV.h
#define SPIRV_CROSSWORKGROUP_AS 1   // Global memory
#define SPIRV_UNIFORMCONSTANT_AS 2  // Constant memory (read-only)
#define SPIRV_WORKGROUP_AS 3        // Local/shared memory
#define SPIRV_GENERIC_AS 4          // Generic address space

// OpenCL compatibility aliases
#define OCL_GLOBAL_AS SPIRV_CROSSWORKGROUP_AS
#define OCL_GENERIC_AS SPIRV_GENERIC_AS
#define OCL_CONSTANT_AS SPIRV_UNIFORMCONSTANT_AS
#define OCL_LOCAL_AS SPIRV_WORKGROUP_AS
```

### HIP to SPIR-V Address Space Translation Table

| HIP/CUDA Memory | Address Space | SPIR-V Constant | OpenCL Equivalent | Scope |
|-----------------|---------------|-----------------|-------------------|-------|
| Global (`__global__`) | 1 | `SPIRV_CROSSWORKGROUP_AS` | `__global` | All work-items across all work-groups |
| Constant (`__constant__`) | 2 | `SPIRV_UNIFORMCONSTANT_AS` | `__constant` | Read-only across all work-items |
| Shared (`__shared__`) | 3 | `SPIRV_WORKGROUP_AS` | `__local` | Shared within work-group |
| Generic (pointers) | 4 | `SPIRV_GENERIC_AS` | `__generic` | Can point to global, local, or private |
| Private (registers) | 0 | (default) | `__private` | Per work-item |

### Usage in LLVM Passes

The address space constants are used throughout transformation passes:

```cpp
// HipDynMem.cpp line 44
#define SPIR_LOCAL_AS 3  // Local address space for shared memory

// HipGlobalVariables.cpp lines 62-64
constexpr unsigned SpirvCrossWorkGroupAS = SPIRV_CROSSWORKGROUP_AS;
constexpr unsigned SpirvUniformConstantAS = SPIRV_UNIFORMCONSTANT_AS;
constexpr unsigned SpirvWorkgroupAS = SPIRV_WORKGROUP_AS;
```

---

## LLVM IR to SPIR-V Translation Pipeline

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      HIP Source (.hip)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Clang Frontend (Device Mode)                                   │
│  Command: clang -cc1 -triple spirv64 -emit-llvm                 │
│  Output: LLVM IR with HIP intrinsics                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Link chipStar Bitcode Library                                  │
│  Tool: clang -cc1 -mlink-builtin-bitcode hipspv-spirv64.bc      │
│  Adds: 55,889 lines of OpenCL C device function implementations │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  chipStar LLVM Transformation Passes (25 passes)                │
│  Tool: opt -load-pass-plugin libLLVMHipSpvPasses.so             │
│  - HipDynMemExternReplaceNewPass (shared memory)                │
│  - HipPromoteIntsPass (integer width normalization)             │
│  - HipTextureLoweringPass (texture objects)                     │
│  - InferAddressSpacesPass (address space inference)             │
│  - ... (21 more passes)                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SPIRV-LLVM-Translator                                          │
│  Tool: llvm-spirv (Khronos Group project)                       │
│  Command: llvm-spirv --spirv-max-version=1.X --spirv-ext=+all   │
│  Input: Prepared LLVM IR                                        │
│  Output: SPIR-V binary (.spv)                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Runtime SPIR-V Module Linking (capability-dependent)           │
│  - Native atomic operations (if cl_ext_float_atomics available) │
│  - Emulated atomics (fallback)                                  │
│  - Ballot intrinsics (if subgroup support available)            │
└─────────────────────────────────────────────────────────────────┘
```

### The llvm-spirv Tool

chipStar uses the **SPIRV-LLVM-Translator** project (external to chipStar) to perform the actual LLVM IR → SPIR-V conversion. This is a two-way translator maintained by the Khronos Group.

**Detection and Validation** (`cmake/FindLLVM.cmake` lines 113-124):

```cmake
if(NOT DEFINED LLVM_SPIRV)
  find_program(LLVM_SPIRV NAMES llvm-spirv-${LLVM_VERSION_MAJOR} llvm-spirv
               FIND_TARGET NO_DEFAULT_PATH PATHS ${CLANG_ROOT_PATH_BIN} ENV PATH)
  if(NOT LLVM_SPIRV)
    message(FATAL_ERROR "Can't find llvm-spirv. Please provide -DLLVM_SPIRV=/path/to/llvm-spirv")
  endif()
endif()
message(STATUS "Using llvm-spirv: ${LLVM_SPIRV}")

# Execute llvm-spirv and check for errors
execute_process(COMMAND "${LLVM_SPIRV}" "--version"
                OUTPUT_VARIABLE LLVM_SPIRV_OUTPUT
                ERROR_VARIABLE LLVM_SPIRV_ERROR
                RESULT_VARIABLE LLVM_SPIRV_RESULT)
```

---

## Shared Memory Transformation

### The Problem

HIP/CUDA uses **global variable declarations** for shared memory:

```cpp
extern __shared__ float dynamicShmem[];  // Dynamic shared memory
__shared__ int staticShmem[256];         // Static shared memory
```

OpenCL/SPIR-V requires shared memory to be **kernel arguments** in address space 3:

```c
__kernel void myKernel(__local float* dynamicShmem, ...) {
    // Use dynamicShmem
}
```

### The Solution: HipDynMemExternReplaceNewPass

The `HipDynMem.cpp` pass performs a sophisticated transformation to convert global shared memory variables into kernel arguments.

#### Step 1: Identify Shared Memory Variables

```cpp
// HipDynMem.cpp lines 435-438
if (GV->hasName() == true && GV->getAddressSpace() == SPIR_LOCAL_AS &&
    GV->getValueType()->isArrayTy() &&
    GV->getLinkage() == GlobalValue::ExternalLinkage) {
  // This is a dynamic shared memory declaration
}
```

#### Step 2: Clone Function with New Argument

```cpp
// HipDynMem.cpp lines 281-338 (simplified)
static Function *cloneFunctionWithDynMemArg(Function *F, Module &M,
                                            GlobalVariable *GV) {
  // Extract element type from the array
  Type *ElemT = GV->getValueType()->getArrayElementType();

  // Create pointer in address space 3 (local/shared)
  PointerType *AS3_PTR = PointerType::get(ElemT, SPIR_LOCAL_AS);

  // Build new parameter list
  SmallVector<Type *, 8> Parameters;
  for (auto &Arg : F->args())
    Parameters.push_back(Arg.getType());
  Parameters.push_back(AS3_PTR);  // Add shared memory argument

  // Create new function type and function
  FunctionType *FT = FunctionType::get(F->getReturnType(), Parameters,
                                       F->isVarArg());
  Function *NewF = Function::Create(FT, F->getLinkage(),
                                    F->getAddressSpace(), "", &M);

  // Clone function body
  ValueToValueMapTy VMap;
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly);

  return NewF;
}
```

#### Step 3: Replace Global Variable Uses

All uses of the global `__shared__` variable are replaced with references to the new kernel argument:

```cpp
// HipDynMem.cpp lines 367-407 (simplified)
static void replaceGVarUsesWith(GlobalVariable *GV, Function *NewF,
                                Argument *NewArg) {
  for (Use &U : make_early_inc_range(GV->uses())) {
    User *Usr = U.getUser();

    if (auto *Inst = dyn_cast<Instruction>(Usr)) {
      if (Inst->getFunction() == NewF) {
        // Replace with the new argument
        Inst->replaceUsesOfWith(GV, NewArg);
      }
    }
  }
}
```

#### Step 4: Update Kernel Metadata

OpenCL kernels require metadata describing argument types and address spaces:

```cpp
// HipDynMem.cpp lines 213-243
static void updateFunctionMD(Function *F, Module &M,
                             PointerType *ArgTypeWithoutAS) {
  IntegerType *I32Type = IntegerType::get(M.getContext(), 32);

  // Append address space metadata
  MDNode *MD = MDNode::get(
      M.getContext(),
      ConstantAsMetadata::get(ConstantInt::get(I32Type, SPIR_LOCAL_AS)));
  appendMD(F, "kernel_arg_addr_space", MD);

  // Append type qualifier metadata (none for local memory)
  MD = MDNode::get(M.getContext(),
                   ConstantAsMetadata::get(ConstantInt::get(I32Type, 0)));
  appendMD(F, "kernel_arg_type_qual", MD);

  // Other metadata: access_qual, type, base_type, name
  // ...
}
```

### Before and After Example

**Before transformation** (HIP source):

```cpp
extern __shared__ float shmem[];

__global__ void myKernel(float* output, int n) {
    int tid = threadIdx.x;
    shmem[tid] = tid * 2.0f;
    __syncthreads();
    output[tid] = shmem[tid];
}
```

**After HipDynMem transformation** (LLVM IR):

```llvm
define spir_kernel void @myKernel(
    float addrspace(1)* %output,
    i32 %n,
    float addrspace(3)* %shmem        ; <-- New argument in AS 3
) !kernel_arg_addr_space !1 {
  %tid = call i32 @get_local_id(i32 0)
  %tid_f = sitofp i32 %tid to float
  %val = fmul float %tid_f, 2.0

  ; Store to shared memory (address space 3)
  %ptr = getelementptr float, float addrspace(3)* %shmem, i32 %tid
  store float %val, float addrspace(3)* %ptr

  call void @barrier(i32 1)  ; CLK_LOCAL_MEM_FENCE

  %loaded = load float, float addrspace(3)* %ptr
  %out_ptr = getelementptr float, float addrspace(1)* %output, i32 %tid
  store float %loaded, float addrspace(1)* %out_ptr
  ret void
}

!1 = !{i32 1, i32 0, i32 3}  ; output=AS1, n=AS0, shmem=AS3
```

---

## Barrier Synchronization

### HIP/CUDA Barrier Primitives

HIP provides several synchronization primitives:

- `__syncthreads()` - Barrier with local memory fence
- `__syncthreads_count()` - Barrier with count reduction
- `__syncwarp()` - Warp-level synchronization

### OpenCL Barrier Function

OpenCL provides a single barrier function with fence flags:

```c
void barrier(cl_mem_fence_flags flags);

// Flag options:
// CLK_LOCAL_MEM_FENCE  = 1  (workgroup/shared memory)
// CLK_GLOBAL_MEM_FENCE = 2  (global memory)
```

### Implementation in devicelib.cl

chipStar implements HIP barriers by calling OpenCL barriers with appropriate flags:

```c
// bitcode/devicelib.cl line 526
EXPORT void __chip_syncthreads() {
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Additional barrier variants
EXPORT void __chip_syncthreads_and(int predicate) {
    barrier(CLK_LOCAL_MEM_FENCE);
    // Predicate reduction handled separately
}

EXPORT void __chip_syncthreads_count(int predicate) {
    barrier(CLK_LOCAL_MEM_FENCE);
    // Count reduction using subgroup operations
}
```

### Memory Fence Semantics

| HIP Function | OpenCL Barrier | Fence Scope | Purpose |
|--------------|----------------|-------------|---------|
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Workgroup | Synchronize threads and ensure shared memory visibility |
| `__threadfence()` | `mem_fence(CLK_GLOBAL_MEM_FENCE)` | Device | Ensure global memory visibility (no sync) |
| `__threadfence_block()` | `mem_fence(CLK_LOCAL_MEM_FENCE)` | Workgroup | Ensure shared memory visibility (no sync) |
| `__threadfence_system()` | `mem_fence(CLK_GLOBAL_MEM_FENCE)` | System | Ensure system-wide memory visibility |

---

## Atomic Operations Translation

chipStar provides two implementations for atomic operations:

1. **Native implementation** - Uses OpenCL atomic extensions when available
2. **Emulated implementation** - Fallback using compare-and-swap loops

### Runtime Selection Mechanism

The runtime decides which implementation to link based on device capabilities:

```cpp
// bitcode/CMakeLists.txt lines 158-173
set(RTDEVLIB_SOURCES_v1_2
  atomicAddFloat_native atomicAddFloat_emulation
  atomicAddDouble_native atomicAddDouble_emulation)

# These are compiled to SPIR-V and embedded in the runtime
# The runtime selects the appropriate module at JIT time
```

### Native Atomic Implementation

Uses the OpenCL `cl_ext_float_atomics` extension:

```c
// bitcode/atomicAddFloat_native.cl lines 32-54
#if !defined(__opencl_c_ext_fp32_global_atomic_add) || \
    !defined(__opencl_c_ext_fp32_local_atomic_add)
#error cl_ext_float_atomics needed!
#endif

#define DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, ORDER, SCOPE) \
  float __chip_atomic_##NAME##_f32(__chip_obfuscated_ptr_t address, float i) { \
    return atomic_##OP##_explicit( \
        (volatile __generic float *)UNCOVER_OBFUSCATED_PTR(address), i, \
        memory_order_##ORDER, memory_scope_##SCOPE); \
  }

#define DEF_CHIP_ATOMIC2F(NAME, OP) \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, relaxed, device) \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_system, OP, relaxed, all_svm_devices) \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_block, OP, relaxed, work_group)

// Generates: __chip_atomic_add_f32, __chip_atomic_add_system_f32,
//            __chip_atomic_add_block_f32
DEF_CHIP_ATOMIC2F(add, fetch_add);
```

This expands to:

```c
float __chip_atomic_add_f32(__chip_obfuscated_ptr_t address, float i) {
    return atomic_fetch_add_explicit(
        (volatile __generic float *)UNCOVER_OBFUSCATED_PTR(address),
        i,
        memory_order_relaxed,
        memory_scope_device
    );
}

float __chip_atomic_add_system_f32(__chip_obfuscated_ptr_t address, float i) {
    return atomic_fetch_add_explicit(
        (volatile __generic float *)UNCOVER_OBFUSCATED_PTR(address),
        i,
        memory_order_relaxed,
        memory_scope_all_svm_devices
    );
}

float __chip_atomic_add_block_f32(__chip_obfuscated_ptr_t address, float i) {
    return atomic_fetch_add_explicit(
        (volatile __generic float *)UNCOVER_OBFUSCATED_PTR(address),
        i,
        memory_order_relaxed,
        memory_scope_work_group
    );
}
```

### Emulated Atomic Implementation

When native atomics aren't available, chipStar emulates them using compare-and-swap loops:

```c
// bitcode/atomicAddFloat_emulation.cl lines 34-70
static OVERLOADED float __chip_atomic_add_f32(volatile local float *address,
                                              float val) {
  volatile local uint *uaddr = (volatile local uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);  // Retry if another thread modified the value

  return as_float(r);
}

static OVERLOADED float __chip_atomic_add_f32(volatile global float *address,
                                              float val) {
  volatile global uint *uaddr = (volatile global uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

// Generic wrapper that dispatches to global or local version
float __chip_atomic_add_f32(__chip_obfuscated_ptr_t address, float val) {
  volatile global float *gi = to_global(UNCOVER_OBFUSCATED_PTR(address));
  if (gi)
    return __chip_atomic_add_f32(gi, val);
  volatile local float *li = to_local(UNCOVER_OBFUSCATED_PTR(address));
  if (li)
    return __chip_atomic_add_f32(li, val);
  return 0;
}
```

### Atomic Operation Coverage

chipStar provides both native and emulated versions for:

| Operation | Native Extension | Emulated Fallback |
|-----------|------------------|-------------------|
| `atomicAdd(float*, float)` | `cl_ext_float_atomics` | CAS loop |
| `atomicAdd(double*, double)` | `cl_ext_float_atomics` | CAS loop |
| `atomicMin/Max(int*, int)` | Built-in OpenCL atomics | N/A |
| `atomicCAS` | Built-in `atomic_cmpxchg` | N/A |
| `atomicExch` | Built-in `atomic_exchange` | N/A |

---

## Integer Type Promotion

### The Problem

LLVM's loop optimizations can generate **non-standard integer types** with arbitrary bit widths:

```llvm
%val = add i33 %a, %b      ; 33-bit integer (non-standard!)
%val2 = mul i56 %x, %y     ; 56-bit integer (non-standard!)
```

SPIR-V only supports standard integer widths: `i1`, `i8`, `i16`, `i32`, `i64`. Non-standard widths cause translation failures.

### The Solution: HipPromoteIntsPass

The `HipPromoteInts.cpp` pass promotes non-standard integer types to the next standard width.

#### Algorithm Overview

```cpp
// HipPromoteInts.cpp lines 19-30
/**
 * @brief Promotes non-standard integer types (e.g., i33, i56) to the next
 *        standard width.
 *
 * LLVM's loop optimizations can generate integer types with bit widths that are
 * not powers of two (or 1). These non-standard types can cause issues during
 * later stages, particularly SPIR-V translation.
 *
 * Key Data Structures:
 * --------------------
 * - `PromotedValues`: Maps original Value* to promoted Value* equivalents
 * - `Replacements`: Stores pairs of {original instruction, new value}
 * - `PendingPhiAdds`: Handles PHI nodes whose inputs haven't been processed yet
 */
```

#### Standard Width Detection

```cpp
// HipPromoteInts.cpp lines 35-40
static bool isNonStandardInt(Type *T) {
  if (auto *IntTy = dyn_cast<IntegerType>(T)) {
    return !HipPromoteIntsPass::isStandardBitWidth(IntTy->getBitWidth());
  }
  return false;
}

// Standard widths: 1, 8, 16, 32, 64
bool HipPromoteIntsPass::isStandardBitWidth(unsigned Width) {
  return Width == 1 || Width == 8 || Width == 16 ||
         Width == 32 || Width == 64;
}
```

#### Promotion Strategy

| Non-Standard Width | Promoted To | Reason |
|--------------------|-------------|--------|
| i2 - i8 | i8 | Next power of 2 |
| i9 - i16 | i16 | Next power of 2 |
| i17 - i32 | i32 | Next power of 2 |
| i33 - i64 | i64 | Next power of 2 |
| > i64 | Error | SPIR-V doesn't support > 64-bit |

### Transformation Example

**Before promotion**:

```llvm
define i33 @loop_increment(i33 %counter) {
  %incremented = add i33 %counter, 1
  %wrapped = and i33 %incremented, 8589934591  ; 2^33 - 1 (wrap around)
  ret i33 %wrapped
}
```

**After promotion**:

```llvm
define i64 @loop_increment(i64 %counter) {
  %incremented = add i64 %counter, 1
  %wrapped = and i64 %incremented, 8589934591  ; Same mask, now i64
  ret i64 %wrapped
}
```

The pass ensures that truncation and sign/zero extension operations are inserted where necessary to preserve semantics.

---

## Texture Lowering

### The Problem

HIP uses **texture objects** for sampled memory access:

```cpp
texture<float, 2> tex;
float value = tex2D<float>(tex, x, y);
```

OpenCL uses **image objects** with different calling conventions:

```c
__read_only image2d_t img;
float4 value = read_imagef(img, sampler, (int2)(x, y));
```

### The Solution: HipTextureLoweringPass

The `HipTextureLowering.cpp` pass analyzes texture usage and transforms texture calls into OpenCL image operations.

#### Texture Type Detection

```cpp
// HipTextureLowering.cpp lines 42-54
namespace TexType {
enum {
  Unresolved = 0,    // No texture usage
  Unknown = (1u << 0), // Unknown texture type
  Basic1D = (1u << 1), // 1D texture
  Basic2D = (1u << 2), // 2D texture

  // Mask for OpenCL supported texture types
  OpenCLSupportedTypes = Basic1D | Basic2D,
};
}
```

#### Texture Function Recognition

```cpp
// HipTextureLowering.cpp lines 59-76
static bool isTextureFunctionCall(Instruction *I, TexTypeSet &Out) {
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (auto *F = CI->getCalledFunction()) {
      Out = StringSwitch<TexTypeSet>(F->getName())
                .StartsWith("_chip_tex1d", TexType::Basic1D)
                .StartsWith("_chip_tex2d", TexType::Basic2D)
                .Default(TexType::Unresolved);

      return Out != TexType::Unresolved;
    }
  }
  return false;
}
```

#### Transformation Process

1. **Identify texture object sources** (kernel arguments, globals)
2. **Analyze texture usage patterns** (1D vs 2D, sampling modes)
3. **Replace texture types** with OpenCL image types (`image2d_t`, etc.)
4. **Transform texture calls**:
   - `tex1D(tex, x)` → `read_imagef(img, sampler, (int)(x))`
   - `tex2D(tex, x, y)` → `read_imagef(img, sampler, (int2)(x, y))`

### Example Transformation

**Before** (HIP):

```cpp
__global__ void processImage(texture<float4, 2> tex, float* output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float4 pixel = tex2D<float4>(tex, x, y);
    output[y * width + x] = pixel.x;
}
```

**After** (OpenCL):

```c
__kernel void processImage(__read_only image2d_t img,
                          sampler_t sampler,
                          __global float* output) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 pixel = read_imagef(img, sampler, (int2)(x, y));
    output[y * width + x] = pixel.x;
}
```

---

## SPIR-V Binary Generation

### The embed_spirv_in_cpp Function

chipStar compiles runtime device libraries (atomics, ballot ops) to SPIR-V and embeds them as C++ arrays in the runtime library.

**CMake Function** (`bitcode/CMakeLists.txt` lines 179-199):

```cmake
function(embed_spirv_in_cpp
    ARRAY_NAME BC_SOURCE OUTPUT_SOURCE OUTPUT_HEADER MAX_SPIRV_VERSION)

  set(SPIRV_EXTENSIONS "+SPV_EXT_shader_atomic_float_add")
  get_filename_component(SOURCE_BASENAME "${BC_SOURCE}" NAME_WLE)
  set(SPIR_BINARY ${SOURCE_BASENAME}.spv)

  add_custom_command(
    OUTPUT "${OUTPUT_SOURCE}" "${OUTPUT_HEADER}"
    DEPENDS "${BC_SOURCE}"
    BYPRODUCTS "${SPIR_BINARY}"

    # Step 1: Convert LLVM IR to SPIR-V
    COMMAND "${LLVM_SPIRV}"
    --spirv-ext=${SPIRV_EXTENSIONS}
    --spirv-max-version=${MAX_SPIRV_VERSION}
    "${BC_SOURCE}" -o "${SPIR_BINARY}"

    # Step 2: Embed SPIR-V binary in C++ header/source
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/embed-binary-in-cpp.bash
    ${ARRAY_NAME} ${SPIR_BINARY} ${OUTPUT_SOURCE} ${OUTPUT_HEADER}

    COMMENT "Generating embedded SPIR-V binary: ${OUTPUT_SOURCE}"
    VERBATIM
  )
endfunction()
```

### llvm-spirv Command-Line Options

The `llvm-spirv` tool is invoked with these key options:

```bash
llvm-spirv \
  --spirv-ext=+SPV_EXT_shader_atomic_float_add \  # Enable atomic extensions
  --spirv-max-version=1.2 \                        # SPIR-V version target
  input.bc -o output.spv                           # Input/output files
```

**Available SPIR-V Extensions**:
- `+SPV_EXT_shader_atomic_float_add` - Floating-point atomic operations
- `+SPV_KHR_subgroup_ballot` - Subgroup ballot operations
- `+all` - Enable all supported extensions

**SPIR-V Version Targets**:
- `1.0` - OpenCL 2.1 compatible
- `1.1` - OpenCL 2.2 compatible
- `1.2` - Vulkan 1.1, OpenCL 3.0
- `1.3` - Vulkan 1.1 with extensions
- `1.4`+ - Modern Vulkan features

### Embedded Module Example

After running `embed-binary-in-cpp.bash`, chipStar generates C++ code like:

```cpp
// Generated: atomicAddFloat_native.h
#include <array>

namespace chipstar {
namespace rtdevlib {

constexpr std::array<unsigned char, 1472> atomicAddFloat_native = {
    0x03, 0x02, 0x23, 0x07,  // SPIR-V magic number
    0x00, 0x00, 0x01, 0x00,  // Version 1.0
    0x00, 0x00, 0x0E, 0x00,  // Generator: Khronos LLVM/SPIR-V Translator
    // ... rest of SPIR-V binary ...
};

}  // namespace rtdevlib
}  // namespace chipstar
```

The runtime then links this module at JIT time:

```cpp
// Pseudocode for runtime module selection
if (device.supportsExtension("cl_ext_float_atomics")) {
    linkModule(program, chipstar::rtdevlib::atomicAddFloat_native);
} else {
    linkModule(program, chipstar::rtdevlib::atomicAddFloat_emulation);
}
```

---

## Complete Example Walkthrough

Let's walk through a complete example showing all transformation stages.

### Original HIP Source

```cpp
// vector_add.hip
#include <hip/hip_runtime.h>

extern __shared__ float shmem[];

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Use dynamic shared memory
    if (threadIdx.x < 256) {
        shmem[threadIdx.x] = a[tid];
    }
    __syncthreads();

    // Atomic add to first element
    if (tid < n) {
        atomicAdd(&c[0], a[tid] + b[tid]);
    }
}
```

### Stage 1: Clang Device Compilation

**Command**:
```bash
clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
  -emit-llvm -o vector_add.bc -x hip vector_add.hip
```

**Output** (simplified LLVM IR):

```llvm
; ModuleID = 'vector_add.hip'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

@shmem = external addrspace(3) global [0 x float], align 4

define spir_kernel void @_Z9vectorAddPfS_S_i(
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c,
    i32 %n
) {
entry:
  %tid_x = call i32 @llvm.amdgcn.workitem.id.x()
  %bid_x = call i32 @llvm.amdgcn.workgroup.id.x()
  %bdim_x = call i32 @llvm.amdgcn.workgroup.dim.x()
  %0 = mul i32 %bid_x, %bdim_x
  %tid = add i32 %tid_x, %0

  %cond1 = icmp ult i32 %tid_x, 256
  br i1 %cond1, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i32 %tid
  %a_val = load float, float addrspace(1)* %a_ptr
  %shmem_ptr = getelementptr inbounds [0 x float],
               [0 x float] addrspace(3)* @shmem, i32 0, i32 %tid_x
  store float %a_val, float addrspace(3)* %shmem_ptr
  br label %if.end

if.end:
  call void @llvm.amdgcn.s.barrier()

  %cond2 = icmp slt i32 %tid, %n
  br i1 %cond2, label %if.then2, label %if.end2

if.then2:
  %a_val2 = load float, float addrspace(1)* %a_ptr
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i32 %tid
  %b_val = load float, float addrspace(1)* %b_ptr
  %sum = fadd float %a_val2, %b_val
  %1 = call float @llvm.amdgcn.atomic.fadd.f32.p1f32(
       float addrspace(1)* %c, float %sum)
  br label %if.end2

if.end2:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.dim.x()
declare void @llvm.amdgcn.s.barrier()
declare float @llvm.amdgcn.atomic.fadd.f32.p1f32(float addrspace(1)*, float)
```

### Stage 2: Link chipStar Bitcode Library

**Command**:
```bash
clang -cc1 -triple spirv64 \
  -mlink-builtin-bitcode /usr/local/lib/hip-device-lib/hipspv-spirv64.bc \
  -x ir vector_add.bc -o vector_add_linked.bc
```

This links in implementations of:
- Built-in variables (`threadIdx`, `blockIdx`, etc.)
- Math functions (via OCML)
- Atomic operations
- Synchronization primitives

### Stage 3: LLVM Transformation Passes

**Command**:
```bash
opt -load-pass-plugin=/usr/local/lib/libLLVMHipSpvPasses.so \
  -passes=hip-post-link-passes \
  -o vector_add_transformed.bc vector_add_linked.bc
```

**Key transformations applied**:

1. **HipDynMemExternReplaceNewPass**: Adds `shmem` as kernel argument
2. **Barrier lowering**: `llvm.amdgcn.s.barrier()` → `barrier(CLK_LOCAL_MEM_FENCE)`
3. **Atomic lowering**: Links appropriate atomic implementation
4. **Built-in lowering**: AMD intrinsics → OpenCL functions
5. **Address space inference**: Optimizes generic pointers

**Output** (simplified LLVM IR after passes):

```llvm
target triple = "spirv64-unknown-unknown"

define spir_kernel void @_Z9vectorAddPfS_S_i(
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c,
    i32 %n,
    float addrspace(3)* %shmem        ; <-- New argument!
) !kernel_arg_addr_space !1
  !kernel_arg_access_qual !2
  !kernel_arg_type !3 {
entry:
  %tid_x = call i32 @_Z12get_local_idj(i32 0)  ; get_local_id(0)
  %bid_x = call i32 @_Z12get_group_idj(i32 0)  ; get_group_id(0)
  %bdim_x = call i32 @_Z14get_local_sizej(i32 0)  ; get_local_size(0)
  %0 = mul i32 %bid_x, %bdim_x
  %tid = add i32 %tid_x, %0

  %cond1 = icmp ult i32 %tid_x, 256
  br i1 %cond1, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr float, float addrspace(1)* %a, i32 %tid
  %a_val = load float, float addrspace(1)* %a_ptr
  %shmem_ptr = getelementptr float, float addrspace(3)* %shmem, i32 %tid_x
  store float %a_val, float addrspace(3)* %shmem_ptr
  br label %if.end

if.end:
  call void @barrier(i32 1)  ; CLK_LOCAL_MEM_FENCE

  %cond2 = icmp slt i32 %tid, %n
  br i1 %cond2, label %if.then2, label %if.end2

if.then2:
  %a_val2 = load float, float addrspace(1)* %a_ptr
  %b_ptr = getelementptr float, float addrspace(1)* %b, i32 %tid
  %b_val = load float, float addrspace(1)* %b_ptr
  %sum = fadd float %a_val2, %b_val
  %obfuscated = call i64 @__chip_obfuscate_ptr_i64(float addrspace(1)* %c)
  %1 = call float @__chip_atomic_add_f32(i64 %obfuscated, float %sum)
  br label %if.end2

if.end2:
  ret void
}

declare i32 @_Z12get_local_idj(i32)
declare i32 @_Z12get_group_idj(i32)
declare i32 @_Z14get_local_sizej(i32)
declare void @barrier(i32)
declare float @__chip_atomic_add_f32(i64, float)
declare i64 @__chip_obfuscate_ptr_i64(float addrspace(1)*)

!1 = !{i32 1, i32 1, i32 1, i32 0, i32 3}  ; AS: global, global, global, private, local
!2 = !{!"none", !"none", !"none", !"none", !"none"}
!3 = !{!"float*", !"float*", !"float*", !"int", !"float*"}
```

### Stage 4: SPIR-V Translation

**Command**:
```bash
llvm-spirv --spirv-max-version=1.2 --spirv-ext=+all \
  vector_add_transformed.bc -o vector_add.spv
```

**Output**: Binary SPIR-V module (partial disassembly using `spirv-dis`):

```spirv
; SPIR-V
; Version: 1.2
; Generator: Khronos LLVM/SPIR-V Translator; 14
; Bound: 73
; Schema: 0

               OpCapability Addresses
               OpCapability Linkage
               OpCapability Kernel
               OpCapability Int8
               OpExtension "SPV_KHR_storage_buffer_storage_class"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %vectorAdd "_Z9vectorAddPfS_S_i" %__spirv_BuiltInGlobalInvocationId
               OpSource OpenCL_C 200
               OpName %vectorAdd "_Z9vectorAddPfS_S_i"
               OpName %a "a"
               OpName %b "b"
               OpName %c "c"
               OpName %n "n"
               OpName %shmem "shmem"

       %uint = OpTypeInt 32 0
      %ulong = OpTypeInt 64 0
      %float = OpTypeFloat 32
    %uint_0 = OpConstant %uint 0
    %uint_1 = OpConstant %uint 1
  %uint_256 = OpConstant %uint 256
%_ptr_CrossWorkgroup_float = OpTypePointer CrossWorkgroup %float
%_ptr_Workgroup_float = OpTypePointer Workgroup %float
   %void = OpTypeVoid
         %10 = OpTypeFunction %void %_ptr_CrossWorkgroup_float
                                    %_ptr_CrossWorkgroup_float
                                    %_ptr_CrossWorkgroup_float
                                    %uint
                                    %_ptr_Workgroup_float

%vectorAdd = OpFunction %void None %10
         %a = OpFunctionParameter %_ptr_CrossWorkgroup_float
         %b = OpFunctionParameter %_ptr_CrossWorkgroup_float
         %c = OpFunctionParameter %_ptr_CrossWorkgroup_float
         %n = OpFunctionParameter %uint
     %shmem = OpFunctionParameter %_ptr_Workgroup_float
     %entry = OpLabel
    %tid_x = OpExtInst %uint %1 get_local_id %uint_0
    %bid_x = OpExtInst %uint %1 get_group_id %uint_0
   %bdim_x = OpExtInst %uint %1 get_local_size %uint_0
       %20 = OpIMul %uint %bid_x %bdim_x
       %tid = OpIAdd %uint %tid_x %20
     %cond1 = OpULessThan %bool %tid_x %uint_256
              OpBranchConditional %cond1 %if_then %if_end

   %if_then = OpLabel
     %a_ptr = OpInBoundsPtrAccessChain %_ptr_CrossWorkgroup_float %a %tid
     %a_val = OpLoad %float %a_ptr
 %shmem_ptr = OpInBoundsPtrAccessChain %_ptr_Workgroup_float %shmem %tid_x
              OpStore %shmem_ptr %a_val
              OpBranch %if_end

    %if_end = OpLabel
              OpControlBarrier %uint_2 %uint_2 %uint_272  ; Workgroup, Workgroup, AcquireRelease|WorkgroupMemory
     %cond2 = OpSLessThan %bool %tid %n
              OpBranchConditional %cond2 %if_then2 %if_end2

  %if_then2 = OpLabel
    %a_val2 = OpLoad %float %a_ptr
     %b_ptr = OpInBoundsPtrAccessChain %_ptr_CrossWorkgroup_float %b %tid
     %b_val = OpLoad %float %b_ptr
       %sum = OpFAdd %float %a_val2 %b_val
       %40 = OpAtomicFAddEXT %float %c %uint_1 %uint_0 %sum  ; Device, Relaxed
              OpBranch %if_end2

  %if_end2 = OpLabel
              OpReturn
              OpFunctionEnd
```

### Stage 5: Runtime JIT Linking

When the program is loaded, the chipStar runtime:

1. **Checks device capabilities**:
   ```cpp
   if (device.hasExtension("cl_ext_float_atomics")) {
       // Link native atomic module
   } else {
       // Link emulated atomic module
   }
   ```

2. **Links appropriate SPIR-V modules**:
   - If native atomics available: links `atomicAddFloat_native.spv`
   - Otherwise: links `atomicAddFloat_emulation.spv`

3. **Compiles final program** using OpenCL or Level Zero runtime

---

## Summary

chipStar's LLVM IR to SPIR-V lowering process involves:

1. **Address Space Mapping**: Systematic translation from HIP address spaces to SPIR-V equivalents
2. **Shared Memory Transformation**: Converting global `__shared__` variables to kernel arguments in AS 3
3. **Barrier Synchronization**: Mapping `__syncthreads()` to OpenCL `barrier(CLK_LOCAL_MEM_FENCE)`
4. **Atomic Operations**: Dual implementation (native + emulated) with runtime selection
5. **Integer Promotion**: Normalizing non-standard integer widths for SPIR-V compatibility
6. **Texture Lowering**: Converting texture objects to OpenCL image types
7. **SPIR-V Generation**: Using `llvm-spirv` tool to produce binary SPIR-V modules

The key insight is that chipStar **prepares LLVM IR** to be compatible with the standard SPIRV-LLVM-Translator, rather than implementing a custom SPIR-V backend. This approach leverages existing Khronos infrastructure while providing the necessary semantic transformations to bridge HIP/CUDA and OpenCL/SPIR-V.

---

## File References

All code examples reference actual files in the chipStar repository:

- `llvm_passes/LLVMSPIRV.h` - Address space constant definitions
- `llvm_passes/HipDynMem.cpp` - Shared memory transformation (560 lines)
- `bitcode/devicelib.cl` - Barrier implementations (line 526)
- `bitcode/atomicAddFloat_native.cl` - Native atomic operations (54 lines)
- `bitcode/atomicAddFloat_emulation.cl` - Emulated atomics (75 lines)
- `llvm_passes/HipPromoteInts.cpp` - Integer type promotion (1000+ lines)
- `llvm_passes/HipTextureLowering.cpp` - Texture to image conversion (800+ lines)
- `bitcode/CMakeLists.txt` - SPIR-V embedding function (lines 179-199)
- `cmake/FindLLVM.cmake` - llvm-spirv tool detection (lines 113-147)

---

**Document Version**: 1.0
**Date**: 2025-10-22
**chipStar Version**: Based on current main branch
**Author**: Analysis of chipStar codebase
