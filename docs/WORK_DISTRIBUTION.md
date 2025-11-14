# Work Distribution Plan: 2 Developers

**Project:** HIP-to-Vortex Compiler Phase 2
**Duration:** 5 weeks
**Status:** Ready to begin

---

## Executive Summary

The Phase 2 compiler work requires implementing **one custom MLIR pass** (~500 lines) that converts GPU dialect operations to Vortex runtime calls. Standard MLIR passes handle all SCF→GPU conversion, eliminating the need for custom loop parallelization or kernel detection code.

The work is split into two balanced modules:
- **Developer A:** Thread Model & Synchronization (~250-300 lines)
- **Developer B:** Memory Operations & Launch Infrastructure (~250-300 lines)

Both developers contribute equally to infrastructure setup, testing, and integration.

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

The system uses **two separate runtime libraries** for different purposes:

### 1. Vortex Runtime Library (Core Runtime)
**Purpose:** Low-level device control and kernel execution
**Location:** Vortex repository `libvortex.so`
**Used by:** Compiled kernels (device code) AND kernel launcher (host code)
**API:**
- **Host-side:** `vx_dev_open()`, `vx_mem_alloc()`, `vx_upload_kernel_bytes()`, `vx_start()`, `vx_ready_wait()`
- **Device-side:** `vx_thread_id()`, `vx_warp_id()`, `vx_barrier()`

### 2. HIP Runtime Library (Optional Binary Compatibility Layer)
**Purpose:** Binary compatibility for pre-compiled x86 host code that was compiled against HIP API
**Location:** `runtime/libhip_vortex.so` (Phase 3 work, optional)
**Used by:** Pre-compiled applications (binaries without source), third-party binary libraries
**API:** `hipMalloc()`, `hipMemcpy()`, `hipLaunchKernel()`, `hipDeviceSynchronize()`, etc.
**Note:** NOT needed when compiling from HIP source - our compiler transforms HIP syntax to Vortex calls directly
**Important:** Cannot provide pre-compiled kernel compatibility - HIP kernels compile to architecture-specific binaries (AMD GCN, NVIDIA PTX) that are incompatible with Vortex RISC-V

### Two Usage Models

#### Model 1: Direct Vortex (Phase 2 - What We're Building)
```
┌─────────────────────────────────────────────────────┐
│ HIP Source (.hip)                                   │
│  __global__ void kernel() { threadIdx.x; }          │
│  int main() {                                       │
│    hipMalloc(&ptr, size);                           │
│    kernel<<<grid, block>>>(...);                    │
│    hipDeviceSynchronize();                          │
│  }                                                  │
└──────────────────┬──────────────────────────────────┘
                   │ Our compiler transforms HIP→Vortex:
                   │  hipMalloc() → vx_mem_alloc()
                   │  kernel<<<>>> → vx_upload/start/wait()
                   │  hipDeviceSynchronize() → vx_ready_wait()
                   │  threadIdx.x → vx_thread_id()
                   ↓
┌─────────────────────────────────────────────────────┐
│ Compiled Binary (links directly to libvortex.so)   │
│  - Host code: Calls vx_* functions directly         │
│  - Device code: Calls vx_* intrinsics               │
└──────────────────┬──────────────────────────────────┘
                   │ Links against & calls
                   ↓
┌─────────────────────────────────────────────────────┐
│ Vortex Runtime Library (libvortex.so)               │
│  Host: vx_dev_open(), vx_upload_kernel_bytes()      │
│  Device: vx_thread_id(), vx_barrier()               │
└──────────────────┬──────────────────────────────────┘
                   │ Controls
                   ↓
┌─────────────────────────────────────────────────────┐
│ Vortex Hardware / Simulator                         │
└─────────────────────────────────────────────────────┘
```

**In Phase 2:** Our compiler transforms ALL HIP constructs (both host API calls like `hipMalloc()` and device syntax like `threadIdx.x`) to Vortex runtime calls. The generated code calls `libvortex.so` directly. No HIP runtime library is needed.

#### Model 2: Binary Compatibility (Phase 3 - Future Work, Optional)
```
┌─────────────────────────────────────────────────────┐
│ Pre-compiled Application (x86 binary)              │
│  - Was compiled against standard HIP API           │
│  - No source code available                        │
│  hipMalloc(&ptr, size);                            │
│  hipLaunchKernel(...);                             │
│  hipDeviceSynchronize();                           │
└──────────────────┬──────────────────────────────────┘
                   │ Links against (at runtime)
                   ↓
┌─────────────────────────────────────────────────────┐
│ HIP Runtime Library (libhip_vortex.so)              │
│  hipMalloc() → vx_mem_alloc()                       │
│  hipLaunchKernel() → vx_upload/start/wait           │
│  - Only for HOST code compatibility                │
│  - Kernels must still be recompiled to .vxbin      │
└──────────────────┬──────────────────────────────────┘
                   │ Calls
                   ↓
┌─────────────────────────────────────────────────────┐
│ Vortex Runtime (libvortex.so)                       │
│  Loads pre-compiled .vxbin kernels                 │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│ Vortex Hardware                                     │
└─────────────────────────────────────────────────────┘
```

**Phase 3 (future):** Build a HIP compatibility library for pre-compiled host binaries.
**Critical limitation:** Kernels must still be recompiled from HIP source to Vortex RISC-V - no architecture-independent HIP kernel binary format exists.

### HIP API Implementation Strategy

HIP uses a **header-based API** approach (same as ROCm and CUDA backends):

**Our Implementation:**
```cpp
// runtime/include/hip/hip_runtime.h (our Vortex backend)

static inline hipError_t hipMalloc(void** ptr, size_t size) {
    return vx_mem_alloc(vx_get_device(), size, ptr);
}

static inline hipError_t hipMemcpy(void* dst, const void* src,
                                    size_t size, hipMemcpyKind kind) {
    if (kind == hipMemcpyHostToDevice)
        return vx_copy_to_dev(vx_get_device(), dst, src, size);
    else if (kind == hipMemcpyDeviceToHost)
        return vx_copy_from_dev(vx_get_device(), dst, src, size);
    // ...
}

static inline hipError_t hipDeviceSynchronize() {
    return vx_ready_wait(vx_get_device(), -1);
}
```

**What this means for compilation:**
1. User includes `<hip/hip_runtime.h>` (our version with Vortex backend)
2. C preprocessor inlines HIP API → Vortex API calls
3. Polygeist sees `vx_*` calls as regular C functions (standard handling)
4. Polygeist **does** handle kernel launch syntax `<<<>>>` via `--cuda-lower` flag
5. GPUToVortexLLVM pass handles kernel constructs (`threadIdx`, `blockIdx`, `__syncthreads()`)

**Complete compilation flow:**
```
user_code.hip
    ↓ [C Preprocessor]
Expanded source (vx_* calls visible, <<<>>> preserved)
    ↓ [Polygeist with --cuda-lower]
MLIR with:
  - func.call @vx_mem_alloc (host code)
  - gpu.launch_func (from <<<>>>)
  - gpu.thread_id (from threadIdx in kernel)
    ↓ [Standard MLIR passes]
GPU Dialect
    ↓ [GPUToVortexLLVM - our custom pass]
LLVM Dialect with vx_* intrinsics
```

This is exactly how HIP works with ROCm and CUDA - backend-specific headers provide the implementation.

### What Our Compiler Pass Generates

The **GPUToVortexLLVM pass** generates two types of code:

#### Host-Side Code (Kernel Launch)
Converts `gpu.launch_func` to Vortex runtime calls:
```mlir
// Input: GPU Dialect
gpu.launch_func @kernels::@myKernel
    blocks in (%c256, %c1, %c1)
    threads in (%c256, %c1, %c1)
    args(%arg0 : memref<?xf32>)

// Output: LLVM Dialect with Vortex runtime calls
llvm.call @vx_upload_kernel_bytes(%device, %kernel_binary, %size)
llvm.call @vx_start(%device)
llvm.call @vx_ready_wait(%device, %timeout)
```

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

---

## Developer A: Thread Model & Kernel Launch

**Estimated Time:** 2-3 weeks
**Estimated LOC:** ~250-300 lines + tests

### Responsibilities

#### 1. Thread & Block ID Mapping (~100-150 lines)

**Convert GPU dialect thread operations to Vortex runtime calls:**

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

#### 2. Synchronization Primitives (~50-75 lines)

**Convert GPU synchronization to Vortex barriers:**

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

#### 3. Kernel Launch Infrastructure (~75-100 lines)

**Convert `gpu.launch_func` to Vortex kernel execution sequence:**

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
call @vx_start(device)

// 4. Wait for completion
call @vx_ready_wait(device, timeout)
```

**Vortex Host-Side API (for kernel launch):**
- `vx_upload_kernel_bytes(device, kernel_data, size)` - Upload kernel to device memory
- `vx_start(device)` - Start kernel execution
- `vx_ready_wait(device, timeout)` - Wait for kernel completion
- `vx_copy_to_dev(device, dest, src, size)` - Copy arguments to device

**Implementation Details:**
- Extract kernel binary reference from `gpu.module`
- Calculate grid/block dimensions for Vortex (warp/core mapping)
- Package kernel arguments (coordinate with Developer B on argument structure)
- Generate complete launch sequence
- Handle launch configuration (grid, block sizes)

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
**Estimated LOC:** ~250-300 lines + tests

### Responsibilities

**Note:** HIP host API calls (`hipMalloc`, `hipMemcpy`, etc.) are handled by header files, NOT by this compiler pass. This pass only handles kernel-side memory operations.

#### 1. Memory Operations (~150-200 lines)

**Convert GPU dialect memory operations to Vortex API:**

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

#### 2. Kernel Launch Infrastructure (~50-100 lines)

**Convert GPU kernel launch to Vortex invocation:**

```mlir
gpu.launch blocks(%bx, %by, %bz) threads(%tx, %ty, %tz) {
  // kernel body
}

→

// Set up kernel arguments structure
call @vx_upload_kernel_bytes(kernel_binary)
call @vx_start(num_warps, num_threads)
call @vx_ready_wait(device)
```

**Vortex API Functions:**
- `vx_upload_kernel_bytes(kernel_data, size)` - Upload kernel to device
- `vx_start(device)` - Start kernel execution
- `vx_ready_wait(device, timeout)` - Wait for completion

**Implementation Details:**
- Extract kernel body as separate function
- Package kernel arguments into Vortex argument structure
- Map grid/block dimensions to Vortex warp/thread counts
- Handle kernel entry/exit sequences
- Set up proper function calling conventions

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

## Shared Work Schedule

### Week 1: Setup & HIP Testing

**Monday-Tuesday: Phase 2A - HIP Syntax Testing (4 hours, pair programming)**
- Test HIP kernel compilation with Polygeist
  ```bash
  cgeist --cuda-lower hip_kernel.hip -S -o hip_kernel.mlir
  ```
- Verify `--cuda-lower` flag works with HIP syntax
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

### Week 4: Metadata Integration

**Monday-Wednesday: Phase 2C - Metadata Extraction (collaborative)**

**Developer A: Extract metadata from MLIR**
- Parse MLIR function signatures
- Extract argument types, sizes, alignments
- Identify pointer vs value arguments
- Generate metadata structures

**Developer B: Kernel Registration Generation**
- Integrate with Phase 1 DWARF-based metadata generator
- Generate kernel registration code
- Create `hipKernelArgumentMetadata` arrays
- Generate module registration functions

**Thursday-Friday: Integration Testing**
- Link thread model + memory model modules
- Test combined pass on simple kernels
- Fix integration issues
- Begin end-to-end testing

### Week 5: End-to-End Testing & Validation

**Monday-Wednesday: Complete Pipeline Testing**
- Test full pipeline: `.hip` → `.vxbin`
- Run all Phase 1 test kernels through compiler:
  - `vecadd_test` - Vector addition
  - `sgemm_test` - Matrix multiplication
  - `dotproduct_test` - Dot product
  - `relu_test` - ReLU activation
  - `fence_test` - Memory fences
  - `cta_test` - Cooperative thread arrays
  - And 7 more tests...
- Compare outputs with Phase 1 manually-written kernels
- Performance validation

**Thursday: Bug Fixes & Optimization**
- Address any test failures
- Performance profiling
- Code optimization
- Documentation updates

**Friday: Final Review & Delivery**
- Final code review
- Documentation completion
- Prepare demo
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

### Week 4 Milestone: Integrated Pass Complete
- ✅ Combined pass compiles successfully
- ✅ Integration tests pass
- ✅ Metadata generation works
- ✅ Simple HIP kernels compile to Vortex binaries

### Week 5 Milestone: Full Pipeline Validated
- ✅ All Phase 1 tests pass with compiled kernels
- ✅ Performance meets or exceeds Phase 1 baselines
- ✅ Documentation complete
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
| **3** | Sync primitives + tests | Launch infrastructure + tests | Code reviews, integration prep |
| **4** | Metadata extraction | Registration code gen | Integration, combined testing |
| **5** | End-to-end testing | End-to-end testing | Full pipeline validation, delivery |

**Total Duration:** 5 weeks
**Total Custom Code:** ~500 lines (250 per developer) + ~200 lines shared infrastructure
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
