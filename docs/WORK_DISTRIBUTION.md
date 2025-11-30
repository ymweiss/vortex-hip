# Work Distribution Plan: 2 Developers

**Project:** HIP-to-Vortex Compiler Phase 2
**Duration:** 5 weeks
**Status:** Phase 2B complete, split compilation validated

---

## Executive Summary

Phase 2 uses a **split compilation model** that separates host and kernel compilation:

- **Host Code:** Compiled with standard g++, links against `libhip_vortex_runtime.a` + `libvortex.so`
- **Kernel Code:** Compiled with Polygeist pipeline, produces `.vxbin` + `.meta.json`

This approach is simpler and more robust than trying to compile both host and kernel through a unified MLIR pipeline.

### What's Complete

✅ **HIP Runtime Library** (`libhip_vortex_runtime.a`)
- All HIP memory APIs (hipMalloc, hipFree, hipMemcpy, hipMemset)
- All HIP device APIs (hipSetDevice, hipGetDeviceProperties, hipDeviceSynchronize)
- Error handling APIs (hipGetErrorString, hipGetLastError)
- Kernel launch infrastructure (hipLaunchKernelGGL, hipRegisterKernel)

✅ **Kernel Compilation Pipeline**
- Polygeist converts HIP kernels to MLIR GPU dialect
- ConvertGPUToVortex pass lowers to Vortex intrinsics
- 21/22 kernel-only files convert successfully

✅ **Split Compilation Validated**
- Host compiles with standard g++
- test_native_kernel passes with Vortex vecadd kernel

### Remaining Work

📋 **Kernel Metadata Emission**
- Output `.meta.json` alongside `.vxbin` during kernel compilation
- Runtime parses metadata for automatic argument marshaling

### Kernel-Side vs Host-Side Work

**Legend:**
- 🔵 **KERNEL-SIDE** = Device code (runs on Vortex RISC-V GPU cores, compiles to .vxbin)
- 🟢 **HOST-SIDE** = Host code (runs on x86 CPU, calls libvortex.so runtime)

**Important:** The compiler generates TWO separate binaries:
1. **Host binary** (x86 ELF) - Contains launch infrastructure, argument packing, runtime API calls
2. **Kernel binary** (.vxbin RISC-V) - Contains thread operations, memory ops, barriers

Each compiler transformation targets one of these two compilation units.

### Split Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIP Source (.hip)                                │
│  ┌─────────────────────────┐     ┌─────────────────────────────┐   │
│  │  __global__ kernel()    │     │  int main() {               │   │
│  │  {                      │     │    hipMalloc(&d_ptr, size); │   │
│  │    // kernel code       │     │    hipLaunchKernelGGL(...); │   │
│  │  }                      │     │    hipDeviceSynchronize();  │   │
│  └─────────────────────────┘     └─────────────────────────────┘   │
└──────────────┬───────────────────────────────┬──────────────────────┘
               │                               │
     ┌─────────┴─────────┐           ┌─────────┴─────────┐
     │  hip_splitter.py  │           │  (keep as-is)     │
     │  Extract kernels  │           │                   │
     └─────────┬─────────┘           └─────────┬─────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│    KERNEL COMPILATION        │   │      HOST COMPILATION            │
│    (Polygeist Pipeline)      │   │      (Standard C++)              │
│    🔵 KERNEL-SIDE            │   │      🟢 HOST-SIDE                │
├──────────────────────────────┤   ├──────────────────────────────────┤
│                              │   │                                  │
│ 1. cgeist --cuda-lower       │   │ 1. g++ -std=c++17                │
│    - HIP → MLIR GPU dialect  │   │    - Standard C++ compilation   │
│                              │   │    - Include hip_vortex_runtime │
│ 2. polygeist-opt             │   │                                  │
│    - GPUToVortex pass        │   │ 2. Link libraries:               │
│    - Thread ID → vx_thread_id│   │    - libhip_vortex_runtime.a    │
│    - Barrier → vx_barrier    │   │    - libvortex.so               │
│    - printf → vx_printf      │   │                                  │
│                              │   │ HIP API → Vortex mapping:        │
│ 3. mlir-translate            │   │    hipMalloc → vx_mem_alloc     │
│    - MLIR → LLVM IR          │   │    hipMemcpy → vx_copy_to_dev   │
│                              │   │    hipLaunchKernelGGL → vx_start│
│ 4. llvm-vortex               │   │    hipDeviceSynchronize →       │
│    - LLVM IR → RISC-V        │   │                  vx_ready_wait  │
│                              │   │                                  │
│ 5. Emit metadata (TODO)      │   │                                  │
│    - kernel.meta.json        │   │                                  │
└──────────────┬───────────────┘   └────────────────┬─────────────────┘
               │                                    │
               ▼                                    ▼
       kernel.vxbin                          host_executable
       kernel.meta.json                      (links libhip_vortex_runtime)
               │                                    │
               └────────────────┬───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   RUNTIME EXECUTION   │
                    │   Host loads kernel   │
                    │   binary at runtime   │
                    │   via hipRegisterKernel│
                    └───────────────────────┘
```

**Key Insight:** Host code does NOT go through MLIR. It uses standard C++ compilation with our HIP runtime library that maps HIP API calls directly to Vortex API calls.

---

## Key Insight: Standard MLIR Handles SCF→GPU

**No custom work needed for SCF→GPU conversion.** The project uses:
- Polygeist for C++/HIP → SCF dialect conversion
- Standard MLIR `--convert-affine-for-to-gpu` pass for SCF → GPU dialect
- **Only custom work:** GPU → Vortex LLVM lowering pass

This architectural decision:
- Leverages mature, tested MLIR infrastructure
- Reduces custom code from ~1000 lines to ~500 lines
- Lowers project risk significantly
- Allows focus on Vortex-specific mappings

---

## Complete Pipeline

```
HIP Source Code (.hip)
    ↓
[Polygeist: cgeist --cuda-lower]
    ↓
MLIR SCF Dialect
    ↓
[Standard MLIR: --convert-affine-for-to-gpu]  ← No custom work needed!
    ↓
MLIR GPU Dialect
    ↓
[Custom Pass: GPUToVortexLLVM]  ← Generates calls to libvortex.so
│   ├─ Developer A: Thread Model & Kernel Launch
│   └─ Developer B: Memory Operations & Argument Marshaling
    ↓
MLIR LLVM Dialect (with vx_* runtime calls)
    ↓
[mlir-translate --mlir-to-llvmir]
    ↓
LLVM IR (.ll)
    ↓
[llvm-vortex]
    ↓
Vortex RISC-V Binary (.vxbin)
```

## Runtime Library Architecture

The split compilation model uses **two runtime libraries**:

### 1. Vortex Runtime Library (Core Runtime)
**Purpose:** Low-level device control and kernel execution
**Location:** `vortex/build/runtime/libvortex.so`
**Used by:** Both host code (via HIP runtime) and kernel code (device intrinsics)
**API:**
- **Host-side:** `vx_dev_open()`, `vx_mem_alloc()`, `vx_upload_kernel_file()`, `vx_start()`, `vx_ready_wait()`
- **Device-side:** `vx_thread_id()`, `vx_warp_id()`, `vx_barrier()`

### 2. HIP Vortex Runtime Library (HIP API Layer) ✅ IMPLEMENTED
**Purpose:** Map HIP API calls to Vortex API for host code
**Location:** `runtime/hip_vortex_runtime/lib/libhip_vortex_runtime.a`
**Used by:** Host code compiled with standard g++
**API:**
```
HIP API                  →  Vortex API
─────────────────────────  ──────────────────────
hipSetDevice()           →  vx_dev_open()
hipGetDeviceProperties() →  vx_dev_caps()
hipMalloc()              →  vx_mem_alloc()
hipFree()                →  vx_mem_free()
hipMemcpy()              →  vx_copy_to_dev() / vx_copy_from_dev()
hipMemset()              →  vx_copy_to_dev() (filled buffer)
hipDeviceSynchronize()   →  vx_ready_wait()
hipGetErrorString()      →  (internal error table)
hipGetLastError()        →  (internal error state)
hipRegisterKernel()      →  vx_upload_kernel_file()
hipLaunchKernelGGL()     →  vx_start()
```

**Build:**
```bash
cd runtime/hip_vortex_runtime && make
# Output: lib/libhip_vortex_runtime.a
```

### Split Compilation Usage Model (Phase 2 - IMPLEMENTED)

```
┌─────────────────────────────────────────────────────┐
│ HIP Source (.hip)                                   │
│  __global__ void kernel() { threadIdx.x; }          │
│  int main() {                                       │
│    hipMalloc(&ptr, size);                           │
│    hipLaunchKernelGGL(kernel, grid, block, ...);   │
│    hipDeviceSynchronize();                          │
│  }                                                  │
└──────────────────┬──────────────────────────────────┘
                   │
     ┌─────────────┴─────────────┐
     │                           │
     ▼                           ▼
┌────────────────────┐   ┌────────────────────────────┐
│ KERNEL EXTRACTION  │   │ HOST COMPILATION           │
│ hip_splitter.py    │   │ g++ -std=c++17             │
└─────────┬──────────┘   │ -I hip_vortex_runtime/inc  │
          │              │ -L hip_vortex_runtime/lib  │
          ▼              │ -lhip_vortex_runtime       │
┌────────────────────┐   │ -L vortex/build/runtime    │
│ KERNEL COMPILATION │   │ -lvortex                   │
│ Polygeist pipeline │   └─────────────┬──────────────┘
│ → kernel.vxbin     │                 │
│ → kernel.meta.json │                 ▼
└─────────┬──────────┘   ┌────────────────────────────┐
          │              │ Host Executable (x86)      │
          │              │ - Links libhip_vortex_runtime│
          │              │ - Links libvortex.so       │
          │              └─────────────┬──────────────┘
          │                            │
          └──────────────┬─────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ RUNTIME EXECUTION    │
              │ 1. hipRegisterKernel │
              │    loads kernel.vxbin│
              │ 2. hipLaunchKernelGGL│
              │    → vx_start()      │
              │ 3. hipDeviceSynchronize│
              │    → vx_ready_wait() │
              └──────────────────────┘
```

**Key Benefits of Split Compilation:**
- Host uses standard g++ - no custom compiler needed
- Kernel compilation isolated - easier debugging
- Runtime library is simple C++ - easy to maintain
- Industry standard approach (CUDA/HIP/OpenCL all do this)

### HIP API Implementation Strategy

**With split compilation, HIP API calls are handled by the runtime library, NOT the compiler.**

The host code is compiled with standard g++ and links against `libhip_vortex_runtime.a`, which provides implementations of all HIP API functions that delegate to Vortex API.

**Host Compilation Flow:**
```bash
# Compile host code with standard g++
g++ -std=c++17 \
    -I runtime/hip_vortex_runtime/include \
    -I vortex/runtime/include \
    -I vortex/build/hw \
    host_code.cpp \
    -L runtime/hip_vortex_runtime/lib -lhip_vortex_runtime \
    -L vortex/build/runtime -lvortex \
    -o host_executable

# Run with library path
LD_LIBRARY_PATH=vortex/build/runtime ./host_executable
```

**Kernel Compilation Flow:**
```bash
# Step 1: Extract kernel from HIP source
python scripts/hip_splitter.py input.hip --output-dir kernels/

# Step 2: Compile kernel with Polygeist
cgeist kernels/input_kernel.hip \
    -I runtime/include/hip \
    --cuda-lower \
    -resource-dir $(clang -print-resource-dir) \
    -S -o kernel.mlir

# Step 3: Lower GPU dialect to Vortex
polygeist-opt kernel.mlir \
    --convert-gpu-to-vortex \
    -o kernel_vortex.mlir

# Step 4: Convert to LLVM IR
mlir-translate kernel_vortex.mlir \
    --mlir-to-llvmir \
    -o kernel.ll

# Step 5: Compile to RISC-V binary
llvm-vortex/bin/clang kernel.ll \
    -target riscv32 \
    -march=rv32imaf \
    -o kernel.vxbin
```

**Key Simplification:** By using split compilation:
- No need to lower HIP API calls in MLIR pass
- Host code uses standard C++ toolchain
- Only kernel code goes through Polygeist
- Runtime library handles all HIP→Vortex mapping

### What Our Compiler Pass Generates (Kernel-Side Only)

With split compilation, the **ConvertGPUToVortex pass** only handles kernel code (device-side). Host code is compiled separately with g++.

#### Device-Side Code (Kernel Body)
Converts GPU dialect operations to Vortex intrinsics:
```mlir
// Input: GPU Dialect
%tid = gpu.thread_id x
%bid = gpu.block_id x
gpu.barrier

// Output: LLVM Dialect with Vortex device calls
%tid = llvm.call @vx_thread_id() : () -> i32
%warp = llvm.call @vx_warp_id() : () -> i32
// Compute block_id from warp_id...
llvm.call @vx_barrier(%bar_id, %num_threads) : (i32, i32) -> ()
```

#### Printf Lowering
Transforms printf calls to vx_printf with core ID injection:
```mlir
// Input: printf("value=%d\n", x)
// Output: vx_printf("cid=%d: value=%d\n", vx_core_id(), x)
```

#### Kernel Metadata Emission (TODO)
The pass should also emit metadata for runtime argument marshaling:
```json
{
  "kernel_name": "vecadd_kernel",
  "arguments": [
    {"name": "src0", "type": "ptr", "size": 8},
    {"name": "src1", "type": "ptr", "size": 8},
    {"name": "dst", "type": "ptr", "size": 8},
    {"name": "n", "type": "u32", "size": 4}
  ]
}
```

---

## Remaining Work Distribution

With split compilation, most host-side work is complete (HIP runtime library). The remaining work focuses on:

1. **Kernel Metadata Emission** - Output argument info during kernel compilation
2. **Runtime Metadata Parsing** - Load and use metadata for argument marshaling
3. **End-to-End Testing** - Validate complete pipeline

---

## Developer A: Kernel Metadata & Testing

**Estimated Time:** 1-2 weeks
**Estimated LOC:** ~150-200 lines + tests
**Scope:** 🔵 **KERNEL-SIDE** (metadata emission in MLIR pass)

### Responsibilities

#### 1. Thread & Block ID Mapping (~100-150 lines) 🔵 **KERNEL-SIDE**

**Convert GPU dialect thread operations to Vortex runtime calls:**
**Location:** Inside kernel functions (device code)
**Target:** RISC-V binary running on Vortex GPU cores

```mlir
// GPU Dialect → Vortex LLVM (Device-Side)
gpu.thread_id x  →  call @vx_thread_id() : () -> i32
gpu.thread_id y  →  call @vx_thread_id() with y offset
gpu.thread_id z  →  call @vx_thread_id() with z offset

gpu.block_id x   →  compute from vx_warp_id() and thread counts
gpu.block_id y   →  compute from vx_warp_id() and grid dimensions
gpu.block_id z   →  compute from vx_warp_id() and grid dimensions

gpu.global_id    →  blockId * blockDim + threadId
```

**Vortex Device-Side API (called from within kernels):**
- `vx_thread_id()` - Get thread ID within warp
- `vx_warp_id()` - Get warp ID
- `vx_num_threads()` - Get total thread count per warp
- `vx_num_warps()` - Get total warp count
- `vx_num_cores()` - Get number of cores

**Implementation Details:**
- Map 3D GPU grid/block model to Vortex's warp-based model
- Handle dimension calculations (x, y, z)
- Compute global thread IDs from local + block IDs
- Handle grid/block dimension queries

**Example Transformation:**
```mlir
// Input: GPU Dialect
gpu.func @kernel() kernel {
    %tid_x = gpu.thread_id x
    %bid_x = gpu.block_id x
    %bdim_x = gpu.block_dim x
    %gid_x = arith.muli %bid_x, %bdim_x : index
    %global_id = arith.addi %gid_x, %tid_x : index
}

// Output: LLVM Dialect with Vortex calls
llvm.func @kernel() {
    // Thread ID (direct call)
    %tid = llvm.call @vx_thread_id() : () -> i32

    // Block ID (computed from warp ID)
    %warp_id = llvm.call @vx_warp_id() : () -> i32
    %num_threads = llvm.call @vx_num_threads() : () -> i32
    %threads_per_block_i32 = llvm.mlir.constant(256 : i32) : i32
    %warps_per_block = llvm.sdiv %threads_per_block_i32, %num_threads : i32
    %bid = llvm.sdiv %warp_id, %warps_per_block : i32

    // Global ID
    %bdim = llvm.mlir.constant(256 : i32) : i32
    %bid_times_bdim = llvm.mul %bid, %bdim : i32
    %global_id = llvm.add %bid_times_bdim, %tid : i32
}
```

#### 2. Synchronization Primitives (~50-75 lines) 🔵 **KERNEL-SIDE**

**Convert GPU synchronization to Vortex barriers:**
**Location:** Inside kernel functions (device code)
**Target:** RISC-V barrier instructions

```mlir
// GPU Dialect → Vortex LLVM
gpu.barrier  →  call @vx_barrier(bar_id, num_threads)
```

**Vortex Barrier API:**
- `vx_barrier(bar_id, num_threads)` - Thread synchronization barrier
- Parameters:
  - `bar_id`: Barrier ID (0-31, hardware supports 32 barriers)
  - `num_threads`: Number of threads to wait for

**Implementation Details:**
- Map GPU barrier semantics to Vortex barrier implementation
- Allocate barrier IDs (track usage, avoid conflicts)
- Calculate correct `num_threads` parameter from block dimensions
- Handle memory fence requirements (implicit in Vortex barrier)

**Example Transformation:**
```mlir
// Input: GPU Dialect
gpu.func @kernel() kernel {
    // ... some work ...
    gpu.barrier
    // ... more work ...
}

// Output: LLVM Dialect
llvm.func @kernel() {
    // ... some work ...

    // Barrier with ID 0, for all threads in block
    %bar_id = llvm.mlir.constant(0 : i32) : i32
    %num_threads = llvm.mlir.constant(256 : i32) : i32  // From block dims
    llvm.call @vx_barrier(%bar_id, %num_threads) : (i32, i32) -> ()

    // ... more work ...
}
```

#### 3. Kernel Launch Infrastructure (~125-150 lines) 🟢 **HOST-SIDE**

**Convert `gpu.launch_func` to Vortex kernel execution sequence:**
**Location:** Host wrapper functions (x86 code)
**Target:** Calls to libvortex.so runtime API

```mlir
// GPU Dialect → Vortex LLVM (Host-Side)
gpu.launch_func @kernels::@myKernel
    blocks in (%bx, %by, %bz)
    threads in (%tx, %ty, %tz)
    args(%arg0, %arg1, ...)

→

// 1. Upload kernel binary to device
call @vx_upload_kernel_bytes(device, kernel_binary, size)

// 2. Set up and copy arguments to device
call @vx_copy_to_dev(device, args_dev_addr, args_struct, args_size)

// 3. Start kernel execution
call @vx_start(device, kernel_buffer, args_buffer)

// 4. Wait for completion
call @vx_ready_wait(device, timeout)
```

**Vortex Host-Side API (for kernel launch):**
- `vx_upload_kernel_bytes(device, kernel_data, size, &buffer)` - Upload kernel to device memory
- `vx_upload_bytes(device, data, size, &buffer)` - Upload argument struct to device memory
- `vx_start(device, kernel_buffer, args_buffer)` - Start kernel execution
- `vx_ready_wait(device, timeout)` - Wait for kernel completion

**Implementation Details:**
- Extract kernel binary reference from `gpu.module`
- Calculate grid/block dimensions for Vortex (warp/core mapping)
- **Extract metadata from kernel arguments** (types, sizes, pointer vs value)
- **Generate argument struct packing code** based on metadata:
  - Allocate struct on stack (llvm.alloca)
  - Pack each argument (llvm.getelementptr + llvm.store)
  - Handle type conversions (memref → pointer, index → i32/i64)
  - Ensure proper alignment and padding
- Generate complete launch sequence (upload kernel, upload args, start, wait)
- Handle launch configuration (grid, block sizes)

#### 3a. Metadata Extraction (~50 lines) 🟢 **HOST-SIDE** - **Required for Kernel Launch**

**Extract and store metadata from `gpu.launch_func` for runtime argument marshaling:**
**Location:** Compiler pass analysis phase
**Target:** MLIR attributes or global constants for host code

```mlir
// Input: GPU Dialect
gpu.launch_func @kernels::@myKernel
    args(%arg0 : memref<?xi32>, %arg1 : i32, %arg2 : i64)

// Extract metadata:
// - arg0: memref<?xi32> → pointer (8 bytes)
// - arg1: i32 → value (4 bytes)
// - arg2: i64 → value (8 bytes)
```

**Metadata Storage Options:**

**Option 1: Function Attributes (Recommended)**
Store metadata as MLIR attributes on the generated launch wrapper function:
```mlir
func.func @launch_wrapper(...) attributes {
  vortex.kernel_name = "_Z13launch_kernelPiii_kernel94555991377168",
  vortex.grid_size = dense<[1, 1, 1]> : tensor<3xi32>,
  vortex.block_size = dense<[256, 1, 1]> : tensor<3xi32>,
  vortex.arg_metadata = [
    {type = "ptr", size = 8},
    {type = "i32", size = 4},
    {type = "i64", size = 8}
  ]
}
```

**Option 2: Global Metadata Constants**
Generate global constant structs containing metadata:
```mlir
llvm.mlir.global constant @kernel_myKernel_metadata : !llvm.struct<...> {
  // kernel_name, grid_dims, block_dims, arg_count, arg_info[]
}
```

**Why Metadata is Required:**

Vortex kernel arguments follow a **struct-based model**:
```c
// Example from vortex/tests/regression/diverge
typedef struct {
  uint32_t num_points;   // 4 bytes
  uint64_t src_addr;     // 8 bytes
  uint64_t dst_addr;     // 8 bytes
} kernel_arg_t;

// Runtime usage:
vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer);
vx_start(device, kernel_buffer, args_buffer);
```

The runtime needs metadata to:
1. **Create correctly-sized argument struct** based on argument types
2. **Pack arguments in correct order** (matching kernel signature)
3. **Distinguish pointers from values** (8-byte addresses vs scalar values)
4. **Handle alignment requirements** (struct padding)
5. **Upload struct to device memory** before kernel launch

**Implementation:**
- Parse argument list from `gpu.launch_func`
- Determine size for each argument type:
  - `memref<*>` → 8 bytes (pointer)
  - `i32` → 4 bytes
  - `i64`, `f64` → 8 bytes
  - `f32` → 4 bytes
- Store metadata as attributes or global constants
- Generate argument packing code that creates struct from metadata

**Example Transformation:**
```mlir
// Input: GPU Dialect (Host-Side)
func.func @host_launch(%arg0: memref<?xf32>) {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index

    gpu.launch_func @kernels::@vectorAdd
        blocks in (%c256, %c1, %c1)
        threads in (%c256, %c1, %c1)
        args(%arg0 : memref<?xf32>)

    return
}

// Output: LLVM Dialect with Vortex runtime calls
llvm.func @host_launch(%arg0: !llvm.ptr) {
    // Get device handle (assume initialized)
    %device = llvm.mlir.addressof @g_vortex_device : !llvm.ptr<ptr>
    %device_h = llvm.load %device : !llvm.ptr<ptr>

    // 1. Upload kernel binary
    %kernel_binary = llvm.mlir.addressof @vectorAdd_vxbin : !llvm.ptr
    %kernel_size = llvm.mlir.constant(8192 : i64) : i64
    %result1 = llvm.call @vx_upload_kernel_bytes(%device_h, %kernel_binary, %kernel_size)
        : (!llvm.ptr, !llvm.ptr, i64) -> i32

    // 2. Set up arguments (coordinate with Developer B)
    %args_struct = llvm.alloca ... // Create arg structure
    // ... populate args_struct ...
    %args_size = llvm.mlir.constant(64 : i64) : i64
    %args_dev_addr = llvm.mlir.constant(0x7FFF0000 : i64) : i64  // Device arg address
    %result2 = llvm.call @vx_copy_to_dev(%device_h, %args_dev_addr, %args_struct, %args_size)
        : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32

    // 3. Start execution
    %result3 = llvm.call @vx_start(%device_h) : (!llvm.ptr) -> i32

    // 4. Wait for completion
    %timeout = llvm.mlir.constant(-1 : i64) : i64  // Infinite timeout
    %result4 = llvm.call @vx_ready_wait(%device_h, %timeout) : (!llvm.ptr, i64) -> i32

    llvm.return
}

// Declare Vortex runtime functions
llvm.func @vx_upload_kernel_bytes(!llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @vx_copy_to_dev(!llvm.ptr, i64, !llvm.ptr, i64) -> i32
llvm.func @vx_start(!llvm.ptr) -> i32
llvm.func @vx_ready_wait(!llvm.ptr, i64) -> i32
```

#### 3. Testing Suite

**Test Coverage:**
- Thread ID mapping correctness (10+ test cases)
  - Single dimension (1D grids/blocks)
  - 2D grids/blocks
  - 3D grids/blocks
  - Edge cases (size 1, maximum size)
- Block ID calculations
- Global ID computations
- Barrier synchronization
  - Simple barriers
  - Multiple barriers in sequence
  - Barriers in loops
  - Barriers with conditionals

**Validation:**
- Compare with Phase 1 manually-written kernel outputs
- Verify thread coordination correctness
- Test with varying grid/block sizes

### Implementation File

`phase2-compiler/GPUToVortexLLVM_ThreadModel.cpp`

---

## Developer B: Memory Operations & Argument Marshaling

**Estimated Time:** 2-3 weeks
**Estimated LOC:** ~300-350 lines + tests
**Scope:** 🔵 **KERNEL-SIDE** (device memory ops) + 🟢 **HOST-SIDE** (HIP API lowering)

### Responsibilities

**Note:** HIP host API calls (`hipMalloc`, `hipMemcpy`, etc.) **ARE part of this compiler pass work** and need to be lowered to Vortex runtime calls. This is currently missing (part of the 30% remaining work).

#### 1. Memory Operations (~150-200 lines) 🔵 **KERNEL-SIDE**

**Convert GPU dialect memory operations to Vortex API:**
**Location:** Inside kernel functions (device code)
**Target:** RISC-V memory instructions with address space attributes

```mlir
// GPU Dialect Memory Operations (kernel-side)
gpu.alloc (shared)  →  __local_mem() allocation or vx_shared_mem_ptr()

// Memory Space Mapping
addrspace(1) (global)  →  Vortex global memory (default)
addrspace(3) (shared)  →  Vortex shared memory (__local__)
addrspace(5) (local)   →  Vortex private/stack memory

// Shared Memory Example
%smem = gpu.alloc() : memref<256xf32, 3>
    ↓
%smem_ptr = llvm.call @__local_mem(i32 1024) : (i32) -> !llvm.ptr<3>
```

**Vortex Device-Side Memory:**
- `__local_mem(size)` - Allocate shared memory (if supported)
- Address space attributes in LLVM IR
- Memory fence operations (if needed beyond barriers)

**Implementation Details:**
- Handle address space conversions in LLVM IR
- Map GPU memory spaces to Vortex equivalents
- Insert appropriate casts and address calculations
- Handle shared memory allocation (via `__local_mem()` or similar)
- Implement load/store operations with correct address spaces

#### 2. HIP Host API Lowering (~100-150 lines) 🟢 **HOST-SIDE** ⚠️ **NOT YET IMPLEMENTED**

**Convert HIP host API calls to Vortex runtime calls:**
**Location:** Host functions (x86 code)
**Target:** Calls to libvortex.so runtime API

```mlir
// Input: MLIR with HIP API calls
func.call @hipMalloc(%ptr_addr, %size) : (!llvm.ptr, i64) -> i32
func.call @hipMemcpy(%dst, %src, %size, %kind) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
func.call @hipDeviceSynchronize() : () -> i32
func.call @hipFree(%ptr) : (!llvm.ptr) -> i32

// Output: LLVM dialect with Vortex calls
%device = llvm.call @vx_get_current_device() : () -> !llvm.ptr
llvm.call @vx_mem_alloc(%device, %size, %flags, %buffer_handle) : (!llvm.ptr, i64, i32, !llvm.ptr) -> i32
llvm.call @vx_copy_to_dev(%device, %dst_addr, %src, %size) : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32
llvm.call @vx_ready_wait(%device, %timeout) : (!llvm.ptr, i64) -> i32
llvm.call @vx_mem_free(%buffer_handle) : (!llvm.ptr) -> i32
```

**HIP API to Vortex API Mapping:**
- `hipMalloc(ptr, size)` → `vx_mem_alloc(device, size, flags, &buffer)` + `vx_mem_address(buffer, ptr)`
- `hipMemcpy(dst, src, size, H2D)` → `vx_copy_to_dev(device, dst_addr, src, size)`
- `hipMemcpy(dst, src, size, D2H)` → `vx_copy_from_dev(device, dst, src_addr, size)`
- `hipDeviceSynchronize()` → `vx_ready_wait(device, -1)`
- `hipFree(ptr)` → `vx_mem_free(buffer)`

**Implementation Details:**
- Detect HIP API function calls by name
- Map hipMemcpyKind enum to vx_copy_to_dev vs vx_copy_from_dev
- Handle device handle management (global or thread-local device)
- Handle buffer handle tracking (map pointers to vx_buffer_h)

**Note:** Current test files don't include HIP API calls, so this needs new test cases.

#### 3. (Reserved for future expansion)

**Note:** Argument marshaling is now part of Developer A's Kernel Launch Infrastructure (section 3) to keep the complete launch sequence unified. Developer B focuses on kernel-side memory operations and host-side HIP API lowering.

#### 3. Testing Suite

**Test Coverage:**
- Memory allocation/deallocation (10+ test cases)
  - Device memory allocation
  - Shared memory allocation
  - Memory leak detection
- Data transfers
  - Host → Device
  - Device → Host
  - Device → Device
  - Large transfers
  - Small transfers
- Address space handling
  - Global memory access
  - Shared memory access
  - Mixed memory accesses
- Kernel launch
  - Simple kernel launches
  - Kernels with arguments
  - Multiple kernel launches
  - Different grid/block configurations

**Validation:**
- Compare data transfer correctness with Phase 1 baselines
- Verify memory operations don't corrupt data
- Test with various data sizes and patterns

### Implementation File

`phase2-compiler/GPUToVortexLLVM_Memory.cpp`

---

## Phase 2C: Post-LLVM Compilation Integration

**Purpose:** Integrate LLVM IR output from Phase 2B with the standard Vortex build system to produce final executable binaries.

**Estimated Time:** 3-4 days (integrated into Week 4-5)
**Estimated LOC:** ~100-150 lines (build system integration)
**Scope:** Build system setup, toolchain configuration, binary packaging

### Overview

Phase 2B generates LLVM IR (.ll files) with embedded Vortex API calls. Phase 2C handles the **post-IR compilation** to produce final binaries that can execute on Vortex hardware/simulator.

**Input from Phase 2B:**
- Host LLVM IR (`.ll`) - Contains vx_mem_alloc, vx_start, vx_ready_wait calls (x86-64 target)
- Kernel LLVM IR (`.ll`) - Contains vx_get_threadIdx, vx_barrier calls (RISC-V target)

**Output from Phase 2C:**
- Host executable (x86 ELF) - Dynamically linked to libvortex.so
- Kernel binary (`.vxbin`) - Vortex binary format, ready for device upload

### Compilation Pipeline Overview

```
Phase 2B LLVM IR (.ll files)
    │
    ├─────────────────────────────────┬─────────────────────────────────┐
    │                                 │                                 │
    ↓                                 ↓                                 ↓
Host LLVM IR (x86-64)        Kernel LLVM IR (RISC-V)         Metadata
- vx_mem_alloc calls         - vx_get_threadIdx calls        - Kernel names
- vx_start calls             - vx_barrier calls              - Argument counts
- vx_ready_wait calls        - TLS variable access           - Grid/block dims
    │                                 │
    ↓                                 ↓
[Phase 2C: Host Compilation]  [Phase 2C: Kernel Compilation]
    │                                 │
    ↓                                 ↓
host_binary (x86 ELF)         kernel.vxbin (Vortex binary)
- Links to libvortex.so       - min_vma/max_vma header
- Calls vx_* runtime API      - Raw RISC-V binary data
    │                                 │
    └─────────────────┬───────────────┘
                      ↓
              Runtime Execution:
              host_binary loads kernel.vxbin
              and executes on Vortex device
```

---

### Phase 2C.1: Host Binary Compilation (~40-50 lines)

**Compile host LLVM IR to x86 executable:**

```bash
# Step 1: Compile LLVM IR to object file
clang++ -std=c++17 -O2 \
    -I$(VORTEX_HOME)/runtime/include \
    -c host.ll -o host.o

# Step 2: Link with Vortex runtime library
clang++ host.o \
    -L$(VORTEX_RT_PATH) -lvortex \
    -lpthread \
    -o executable
```

**Implementation Details:**
- Use standard C++ compiler (clang++ or g++)
- Include Vortex runtime headers: `-I$(VORTEX_HOME)/runtime/include`
- Link dynamically with libvortex.so: `-L$(VORTEX_RT_PATH) -lvortex`
- Ensure vx_* function signatures match vortex.h declarations
- Handle library search paths (LD_LIBRARY_PATH at runtime)

**Build System Integration:**
```cmake
# CMakeLists.txt example
add_executable(${PROJECT_NAME}_host ${HOST_LLVM_IR})
target_include_directories(${PROJECT_NAME}_host PRIVATE ${VORTEX_RUNTIME_INCLUDE})
target_link_libraries(${PROJECT_NAME}_host vortex pthread)
target_link_directories(${PROJECT_NAME}_host PRIVATE ${VORTEX_RT_PATH})
```

---

### Phase 2C.2: Kernel Binary Compilation (~50-70 lines)

**Compile kernel LLVM IR to RISC-V binary:**

```bash
# Step 1: Compile LLVM IR to RISC-V object file
$(LLVM_VORTEX)/bin/clang++ \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -Xclang -target-feature -Xclang +vortex \
    --sysroot=$(RISCV_SYSROOT) \
    --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) \
    -O3 -mcmodel=medany \
    -fno-rtti -fno-exceptions \
    -fdata-sections -ffunction-sections \
    -nostartfiles -nostdlib \
    -I$(VORTEX_HOME)/kernel/include \
    -c kernel.ll -o kernel.o

# Step 2: Link with Vortex kernel runtime (libvortex.a)
$(LLVM_VORTEX)/bin/clang++ \
    kernel.o \
    -Wl,-Bstatic,--gc-sections \
    -T$(VORTEX_HOME)/kernel/scripts/link32.ld \
    --defsym=STARTUP_ADDR=0x80000000 \
    $(VORTEX_KN_PATH)/libvortex.a \
    -L$(LIBC_VORTEX)/lib -lm -lc \
    $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv32.a \
    -o kernel.elf
```

**Critical Flags:**
- `-Xclang -target-feature -Xclang +vortex` - **REQUIRED** for Vortex ISA extensions
- `-march=rv32imaf` - RISC-V 32-bit with integer, multiply, atomic, float
- `-mabi=ilp32f` - ILP32 ABI with single-precision float registers
- `-nostartfiles -nostdlib` - Bare-metal kernel (no OS, no standard startup)
- `-T link32.ld` - Custom linker script defines memory layout
- `--defsym=STARTUP_ADDR=0x80000000` - Kernel entry point address

**libvortex.a Contents:**
The kernel links with `$(VORTEX_KN_PATH)/libvortex.a`, which provides:
- **vx_start.S** - Startup code, TLS initialization, jump to main()
- **vx_spawn.c** - `vx_spawn_threads()` function (maps GPU grid/block to warps)
- **vx_intrinsics.h** - Vortex intrinsics: vx_barrier, vx_get_threadIdx, vx_warp_id
- **vx_syscalls.c** - System call implementations
- **vx_print.c/.S** - Printf support for debugging

**Linker Script (link32.ld):**
Defines memory sections and layout:
```ld
ENTRY(_start)
STARTUP_ADDR = 0x80000000;

SECTIONS {
  . = STARTUP_ADDR;
  .text : { *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) }
  .data : { *(.data .data.*) }
  .bss : { *(.bss .bss.*) }
  ...
}
```

**Build System Integration:**
```cmake
# CMakeLists.txt example
add_executable(${PROJECT_NAME}_kernel ${KERNEL_LLVM_IR})
set_target_properties(${PROJECT_NAME}_kernel PROPERTIES
    COMPILE_FLAGS "-target riscv32-unknown-elf -march=rv32imaf -Xclang -target-feature -Xclang +vortex"
    LINK_FLAGS "-T${VORTEX_HOME}/kernel/scripts/link32.ld --defsym=STARTUP_ADDR=0x80000000")
target_link_libraries(${PROJECT_NAME}_kernel ${VORTEX_KN_PATH}/libvortex.a m c)
```

---

### Phase 2C.3: Binary Packaging (~20-30 lines)

**Convert kernel.elf to Vortex binary format (.vxbin):**

```bash
# Generate disassembly (optional, for debugging)
$(LLVM_VORTEX)/bin/llvm-objdump -D kernel.elf > kernel.dump

# Convert to .vxbin format
OBJCOPY=$(LLVM_VORTEX)/bin/llvm-objcopy \
    $(VORTEX_HOME)/kernel/scripts/vxbin.py \
    kernel.elf kernel.vxbin
```

**vxbin.py Script:**
- **Location:** `vortex/kernel/scripts/vxbin.py` (already exists, no modifications needed)
- **Purpose:** Converts RISC-V ELF to Vortex binary format

**vxbin Format:**
```
Offset  Size    Content
────────────────────────────────
0       8       min_vma (minimum virtual address, little-endian uint64)
8       8       max_vma (maximum virtual address, little-endian uint64)
16      N       Raw binary data from ELF LOAD segments
```

**How vxbin.py Works:**
1. Reads kernel.elf and extracts LOAD segments using `readelf`
2. Determines min_vma and max_vma from segment addresses
3. Extracts raw binary using `llvm-objcopy -O binary`
4. Packages as: `[min_vma][max_vma][binary_data]`

**Build System Integration:**
```cmake
# Add custom command to generate .vxbin
add_custom_command(
    OUTPUT ${PROJECT_NAME}.vxbin
    COMMAND OBJCOPY=${LLVM_OBJCOPY} ${VXBIN_SCRIPT}
            ${PROJECT_NAME}_kernel ${PROJECT_NAME}.vxbin
    DEPENDS ${PROJECT_NAME}_kernel
    COMMENT "Generating Vortex binary format"
)
```

---

### Phase 2C.4: Build System Integration (~30-50 lines)

**Toolchain Configuration:**

```cmake
# CMakeLists.txt - Toolchain setup
set(VORTEX_HOME "$ENV{VORTEX_HOME}" CACHE PATH "Vortex home directory")
set(LLVM_VORTEX "$ENV{LLVM_VORTEX}" CACHE PATH "LLVM-Vortex toolchain path")
set(RISCV_TOOLCHAIN_PATH "$ENV{RISCV_TOOLCHAIN_PATH}" CACHE PATH "RISC-V GNU toolchain")

# Architecture selection
set(XLEN 32 CACHE STRING "RISC-V XLEN (32 or 64)")
if(XLEN EQUAL 64)
    set(ARCH_FLAGS "-march=rv64imafd -mabi=lp64d")
    set(STARTUP_ADDR "0x180000000")
else()
    set(ARCH_FLAGS "-march=rv32imaf -mabi=ilp32f")
    set(STARTUP_ADDR "0x80000000")
endif()

# Vortex runtime library paths
set(VORTEX_RT_PATH "${VORTEX_HOME}/runtime/lib")
set(VORTEX_KN_PATH "${VORTEX_HOME}/kernel")
```

**Makefile Alternative:**
```makefile
# Makefile - Follows vortex/tests/regression/common.mk pattern
include $(VORTEX_HOME)/build/config.mk

# Compilation rules
%.o: %.ll
	$(VX_CXX) $(VX_CFLAGS) -c $< -o $@

kernel.elf: kernel.o
	$(VX_CXX) $^ $(VX_LDFLAGS) -o $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

# Host compilation
host: host.ll
	$(CXX) $(CXXFLAGS) $< -L$(VORTEX_RT_PATH) -lvortex -o $@
```

---

### Phase 2C.5: Integration with Existing Infrastructure

**Reuse Existing Vortex Components:**
- ✅ **vxbin.py** - Already exists, no modifications needed
- ✅ **link32.ld / link64.ld** - Linker scripts already provided
- ✅ **libvortex.a** - Kernel runtime already built
- ✅ **libvortex.so** - Host runtime already built
- ✅ **common.mk** - Build patterns from vortex/tests/regression/

**Follow Proven Patterns:**
Reference implementation: `vortex/tests/regression/vecadd/`
- Host code: main.cpp (uses vortex.h API)
- Kernel code: kernel.cpp (uses vx_spawn.h)
- Build system: Makefile includes ../common.mk
- Execution: `LD_LIBRARY_PATH=$(VORTEX_RT_PATH) VORTEX_DRIVER=simx ./vecadd`

**Environment Variables:**
```bash
export VORTEX_HOME=/path/to/vortex
export LLVM_VORTEX=/path/to/llvm-vortex
export RISCV_TOOLCHAIN_PATH=/path/to/riscv-gnu-toolchain
export VORTEX_DRIVER=simx  # or rtlsim, fpga
```

---

### Validation and Testing

**Phase 2C Validation:**
1. **Binary format check:** Generated .vxbin matches standard Vortex format
2. **Symbol resolution:** All vx_* calls resolve correctly
3. **Disassembly review:** kernel.dump shows valid RISC-V instructions
4. **Runtime loading:** vx_upload_kernel_file() successfully loads .vxbin
5. **Execution test:** Kernel runs and produces correct results

**Integration Test Cases:**
- Compile Phase 2B LLVM IR for vecadd test
- Compare binary output with standard Vortex vecadd.vxbin
- Run through simulator (VORTEX_DRIVER=simx)
- Verify results match expected output
- Test with all Phase 1 test kernels

**Success Criteria:**
- ✅ Host binary links and executes without errors
- ✅ Kernel .vxbin format validated (correct header, VMA range)
- ✅ Generated binaries run on Vortex simulator
- ✅ Output matches Phase 1 manually-written kernels
- ✅ Build system integrates with existing Vortex infrastructure

---

### Developer Assignment

**Collaborative Work (Both Developers) - Week 4-5**

**Week 4: Phase 2C Setup and Integration**
- Monday-Tuesday: Toolchain configuration and testing
  - Configure LLVM_VORTEX, RISCV_TOOLCHAIN_PATH
  - Test host compilation (compile + link with libvortex.so)
  - Test kernel compilation (RISC-V + vxbin.py packaging)
  - Validate against existing vecadd example
- Wednesday-Friday: Build system integration
  - Create CMakeLists.txt or Makefile rules
  - Handle RV32 vs RV64 architecture selection
  - Integrate with Phase 2B LLVM IR output
  - Test incremental builds

**Week 5: End-to-End Pipeline Validation**
- Complete pipeline: `.hip` → MLIR → LLVM IR → **Phase 2C** → binaries → execution
- Run all Phase 1 test kernels through complete flow
- Performance validation (compare with Phase 1 baselines)
- Documentation and build system cleanup

---

### Implementation Files

**Build System:**
- `phase2-compiler/CMakeLists.txt` - CMake build rules (or Makefile alternative)
- `phase2-compiler/toolchain.cmake` - Toolchain configuration
- `scripts/compile-hip-binary.sh` - Wrapper script for complete compilation

**No Modifications Needed:**
- ✅ `vortex/kernel/scripts/vxbin.py` - Already functional
- ✅ `vortex/kernel/scripts/link32.ld` - Already correct
- ✅ `vortex/kernel/libvortex.a` - Already built
- ✅ `vortex/runtime/libvortex.so` - Already built

---

## Shared Work Schedule

### Week 1: Setup & HIP Testing

**Monday: Create HIP Runtime Header (4 hours, collaborative)**
- Create `runtime/include/hip/hip_runtime.h`
- Implement inline functions for HIP API → Vortex API mapping:
  ```cpp
  // Essential functions needed for testing
  static inline hipError_t hipMalloc(void** ptr, size_t size);
  static inline hipError_t hipFree(void* ptr);
  static inline hipError_t hipMemcpy(void* dst, const void* src,
                                      size_t size, hipMemcpyKind kind);
  static inline hipError_t hipDeviceSynchronize();
  ```
- Define HIP types and constants (hipError_t, hipMemcpyKind, etc.)
- Test header compiles and links with existing Phase 1 runtime

**Tuesday: Phase 2A - HIP Syntax Testing (4 hours, pair programming)**
- Test HIP kernel compilation with Polygeist
  ```bash
  cgeist --cuda-lower hip_kernel.hip \
      -I runtime/include \
      -resource-dir $(clang -print-resource-dir) \
      -S -o hip_kernel.mlir
  ```
- Verify `--cuda-lower` flag works with HIP syntax
- Validate our `hip_runtime.h` header is correctly included
- Validate standard MLIR passes work: `--convert-affine-for-to-gpu`
- Document findings and required flags
- Create example test case

**Wednesday-Friday: Pass Infrastructure Setup (~100 lines, collaborative)**
- Create base `GPUToVortexLLVM` pass class structure
- Set up MLIR dialect conversion framework
- Define common helper functions:
  - Vortex function declaration insertion
  - Type conversion helpers
  - Debug/logging utilities
- Create build system integration (CMakeLists.txt)
- Set up testing infrastructure
- Each developer branches for their module

### Weeks 2-3: Independent Implementation

**Developer A:**
- Implement thread model operations
- Write unit tests for thread ID mappings
- Test synchronization primitives
- Code reviews with Developer B

**Developer B:**
- Implement memory operations
- Write unit tests for memory transfers
- Test kernel launch infrastructure
- Code reviews with Developer A

**Shared:**
- Daily standups (15 min)
- Code reviews (1-2 hours/week)
- Integration check-ins (Friday afternoons)

### Week 4: Integration & Phase 2C Setup

**Monday-Tuesday: Phase 2C - Toolchain Setup (collaborative)**
- Configure llvm-vortex and RISC-V toolchain paths
- Test host compilation (x86 + libvortex.so linking)
- Test kernel compilation (RISC-V + vxbin.py)
- Validate against vortex/tests/regression/vecadd example

**Wednesday: Metadata Extraction (collaborative)**

**Developer A: Extract metadata from MLIR**
- Parse MLIR function signatures from gpu.launch_func
- Extract argument types, sizes, alignments
- Identify pointer vs value arguments
- Store metadata as MLIR attributes

**Developer B: Build System Integration**
- Create CMakeLists.txt or Makefile for Phase 2C
- Integrate Phase 2B LLVM IR output with compilation pipeline
- Handle RV32 vs RV64 architecture selection
- Add vxbin.py invocation rules

**Thursday-Friday: Integration Testing**
- Link thread model + memory model modules
- Test combined pass on simple kernels
- Compile LLVM IR to binaries via Phase 2C
- Fix integration issues
- Begin end-to-end testing

### Week 5: End-to-End Testing & Validation

**Monday-Wednesday: Complete Pipeline Testing**
- Test full pipeline: `.hip` → MLIR → LLVM IR → **Phase 2C binaries** → execution
- Validate Phase 2C binary generation:
  - Host binary links correctly with libvortex.so
  - Kernel .vxbin format matches standard Vortex binaries
  - Disassembly shows valid RISC-V instructions
- Run all Phase 1 test kernels through complete compiler:
  - `vecadd_test` - Vector addition
  - `sgemm_test` - Matrix multiplication
  - `dotproduct_test` - Dot product
  - `relu_test` - ReLU activation
  - `fence_test` - Memory fences
  - `cta_test` - Cooperative thread arrays
  - And 7 more tests...
- Execute on Vortex simulator (VORTEX_DRIVER=simx)
- Compare outputs with Phase 1 manually-written kernels
- Performance validation against Phase 1 baselines

**Thursday: Bug Fixes & Optimization**
- Address any test failures
- Fix Phase 2C build system issues
- Performance profiling and optimization
- Documentation updates (Phase 2C usage guide)

**Friday: Final Review & Delivery**
- Final code review (Phase 2B + Phase 2C)
- Documentation completion
- Prepare demo (.hip source → execution)
- Project retrospective

---

## Why This Split is Balanced

### 1. Equal Complexity
- Thread model requires understanding Vortex thread/warp model (~250 lines)
- Memory model requires understanding Vortex memory API (~250 lines)
- Both involve MLIR dialect conversion patterns
- Similar learning curves for both developers

### 2. Clear Boundaries
- Thread operations completely independent from memory operations
- Clean interface via GPU dialect
- Minimal shared state or dependencies
- Easy to develop in parallel

### 3. Equal Testing Burden
- Both require ~10 test cases
- Both need unit tests + integration tests
- Similar validation complexity
- Both contribute to end-to-end testing

### 4. Independent Development Timeline
- Can work in parallel for 3 weeks (weeks 2-4)
- Minimal merge conflicts (separate files)
- Clear integration point (week 4)
- Collaborative work in weeks 1 and 5

### 5. Shared Learning
- Both learn MLIR pass infrastructure together (week 1)
- Both understand complete pipeline
- Both participate in testing and debugging
- Knowledge transfer through code reviews

### 6. Risk Distribution
- If one module encounters issues, other can proceed
- Both modules equally critical (neither blocks the other)
- Parallel development reduces timeline risk
- Collaborative integration ensures quality

---

## Success Criteria

### Week 3 Milestone: Individual Modules Complete
- ✅ Thread model passes all unit tests
- ✅ Memory model passes all unit tests
- ✅ Both modules independently validated
- ✅ Code reviewed and documented

### Week 4 Milestone: Integrated Pass + Phase 2C Setup Complete
- ✅ Combined Phase 2B pass compiles successfully
- ✅ Integration tests pass (Phase 2B generates valid LLVM IR)
- ✅ Metadata generation works
- ✅ Phase 2C toolchain configured (llvm-vortex, RISC-V toolchain)
- ✅ Phase 2C can compile host LLVM IR → x86 executable
- ✅ Phase 2C can compile kernel LLVM IR → .vxbin binary
- ✅ Simple HIP kernels compile through complete pipeline (.hip → .vxbin)

### Week 5 Milestone: Full Pipeline Validated
- ✅ All Phase 1 tests pass with Phase 2C-compiled kernels
- ✅ Generated binaries execute correctly on Vortex simulator
- ✅ Binary format validated (.vxbin matches standard Vortex format)
- ✅ Performance meets or exceeds Phase 1 baselines
- ✅ Documentation complete (Phase 2B + Phase 2C)
- ✅ Build system integrated with Vortex infrastructure
- ✅ Code ready for production use

---

## Tools & Resources

### Development Environment
- **MLIR/LLVM:** Polygeist build includes necessary MLIR libraries
- **Build System:** CMake (integrated with Polygeist build)
- **Testing:** MLIR FileCheck tests + custom test runner
- **Debugging:** `mlir-opt` for intermediate IR inspection

### Documentation References
- **MLIR Conversion Patterns:** [mlir.llvm.org/docs/Dialects/GPU](https://mlir.llvm.org/docs/Dialects/GPU/)
- **Vortex Runtime API:** `vortex/runtime/include/vortex.h`
- **Phase 1 Tests:** `tests/*/kernel.cpp` - Reference implementations
- **Project Docs:** `docs/implementation/COMPILER_INFRASTRUCTURE.md`

### Communication
- Daily standups (15 min)
- Weekly planning meetings (1 hour)
- Continuous code reviews via GitHub PRs
- Shared documentation updates

---

## Estimated Timeline Summary

| Week | Developer A | Developer B | Shared Work |
|------|-------------|-------------|-------------|
| **1** | Thread model design | Memory model design | HIP testing (4h), Infrastructure setup |
| **2** | Thread ID implementation | Memory ops implementation | Code reviews, standup |
| **3** | Sync primitives + Launch infrastructure + Argument marshaling + tests | HIP API lowering + tests | Code reviews, integration prep |
| **4** | Metadata extraction + integration | Build system integration (Phase 2C) | Phase 2C toolchain setup, combined testing |
| **5** | End-to-end testing + Phase 2C validation | End-to-end testing + Phase 2C validation | Full pipeline validation (.hip → .vxbin → execution) |

**Total Duration:** 5 weeks
**Total Custom Code:** ~650-750 lines (~350-400 per developer) + ~200 lines shared infrastructure
**Total Testing Code:** ~400 lines (200 per developer)

---

## Next Steps

1. **Review and approve this plan** (both developers + lead)
2. **Set up development branches** (`feature/thread-model`, `feature/memory-model`)
3. **Schedule Week 1 kickoff** (HIP testing + infrastructure setup)
4. **Create tracking issues** (one per major task)
5. **Begin Week 1 work** (Phase 2A: HIP testing)

---

## Notes

- This plan assumes Polygeist is already built and validated (✅ complete)
- Standard MLIR passes are available and tested (✅ verified)
- Phase 1 runtime and tests are working (✅ complete)
- Vortex GPU and llvm-vortex are available as submodules (✅ ready)

**Key Risk Mitigation:** By using standard MLIR passes for SCF→GPU conversion, we've eliminated the highest-risk component of the original plan. The remaining work is straightforward dialect conversion with clear Vortex API mappings.

---

## Kernel-Side vs Host-Side Work Summary

### Developer A Work Breakdown

| Task | Side | LOC | Description |
|------|------|-----|-------------|
| Thread ID mapping | 🔵 KERNEL | ~100-150 | Convert gpu.thread_id/block_id to vx_thread_id()/vx_warp_id() |
| Synchronization | 🔵 KERNEL | ~50-75 | Convert gpu.barrier to vx_barrier() |
| Kernel launch | 🟢 HOST | ~125-150 | Generate vx_upload/start/wait sequence + argument packing |
| Metadata extraction | 🟢 HOST | ~50 | Extract argument metadata for marshaling |
| **TOTAL** | **Mixed** | **~350-400** | **50% kernel, 50% host** |

### Developer B Work Breakdown

| Task | Side | LOC | Description | Status |
|------|------|-----|-------------|--------|
| Memory operations | 🔵 KERNEL | ~150-200 | Address spaces, shared memory allocation | ✓ |
| HIP API lowering | 🟢 HOST | ~100-150 | Convert hipMalloc/hipMemcpy/etc to vx_* calls | ⚠️ TODO |
| **TOTAL** | **Mixed** | **~300-350** | **50% kernel, 50% host** | **60% done** |

### Overall Work Distribution

**Total Compiler Pass:** ~650-750 lines
- **Kernel-side (🔵):** ~300-400 lines (45%) - Runs on Vortex GPU, compiles to RISC-V .vxbin
- **Host-side (🟢):** ~350-450 lines (55%) - Runs on x86 CPU, calls libvortex.so

**Current Implementation Status:** ~70% complete
- ✓ Kernel-side operations (thread IDs, barriers, metadata extraction)
- ✓ Host-side kernel launch infrastructure (partial)
- ⚠️ HIP host API lowering (hipMalloc, hipMemcpy, etc.) - **30% remaining work**

**Both developers work on both kernel-side and host-side code**, ensuring:
- Full understanding of complete compilation pipeline
- Balanced complexity distribution
- Knowledge sharing across host/device boundary
- Better code review quality

**Critical Update:** HIP host API calls (hipMalloc, hipMemcpy, hipDeviceSynchronize) **must be lowered by the compiler pass**. They are NOT handled by header-based inlines. This lowering is part of the 30% remaining work.

This pass handles:
1. **Kernel-side (🔵 45%):** GPU dialect operations → Vortex device intrinsics
   - gpu.thread_id → vx_thread_id()
   - gpu.barrier → vx_barrier()
   - gpu.alloc (shared) → __local_mem()

2. **Host-side (🟢 55%):** Host operations → Vortex runtime API calls
   - gpu.launch_func → vx_upload_kernel_bytes() + vx_start() + vx_ready_wait()
   - func.call @hipMalloc → vx_mem_alloc() ⚠️ **TODO**
   - func.call @hipMemcpy → vx_copy_to_dev() / vx_copy_from_dev() ⚠️ **TODO**
   - func.call @hipDeviceSynchronize → vx_ready_wait() ⚠️ **TODO**
