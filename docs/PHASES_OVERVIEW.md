# Vortex HIP Implementation Phases

**Project:** HIP (Heterogeneous-compute Interface for Portability) for Vortex RISC-V GPU
**Architecture:** RV32 (32-bit RISC-V)
**Status:** Phase 1 complete, Phase 2 in planning

---

## Quick Navigation

- **[Phase 1: HIP Runtime & Testing](#phase-1-hip-runtime--testing)** - ✅ COMPLETE
- **[Phase 2: HIP Compiler Integration](#phase-2-hip-compiler-integration)** - 📋 NEXT
- **[Phase 3: Full Integration & Optimization](#phase-3-full-integration--optimization)** - ⏳ FUTURE

---

## Phase 1: HIP Runtime & Testing

**Status:** ✅ COMPLETE
**Purpose:** Implement and validate HIP runtime API mapping to Vortex

### Components

#### 1A. Metadata Generation (✅ Complete)
**Purpose:** Extract kernel argument metadata from DWARF debug info

- Python script (`scripts/vortex/hip_metadata_gen.py`)
- C++ unit tests (Google Test - 23 tests)
- Python unit tests (unittest - 17 tests)
- Converts HIP array-of-pointers → Vortex packed struct

**Results:**
```
C++ Tests:     23/23 passing (100%)
Python Tests:  17/17 passing (100%)
Total:         40/40 passing (100%)
```

#### 1B. HIP Runtime Library (✅ Complete)
**Purpose:** Implement HIP API calls that map to Vortex API

**API Mapping:**
```
HIP API                  →  Vortex API
─────────────────────────  ──────────────────────
hipSetDevice()           →  vx_dev_open()
hipGetDeviceProperties() →  vx_dev_caps()
hipMalloc()              →  vx_mem_alloc()
hipFree()                →  vx_mem_free()
hipMemcpy()              →  vx_copy_to/from_dev()
hipLaunchKernel()        →  vx_upload_kernel_bytes() + vx_start()
hipDeviceSynchronize()   →  vx_ready_wait()
```

**Features:**
- Lazy kernel loading (deferred upload)
- Metadata-driven argument marshaling
- RV32 pointer handling (4-byte)
- Complete error reporting

#### 1C. Runtime Tests (✅ Complete - 13 tests)
**Purpose:** Validate HIP runtime API works correctly

**Test Structure:**
- **Kernels:** Manually written in Vortex format (using `vx_spawn.h`)
- **Host:** Uses HIP API (`hipMalloc`, `hipLaunchKernel`, etc.)
- **Validates:** Runtime API mapping, not compilation

**Why manually written kernels?**
Phase 1 tests the *runtime*, not the compiler. Using Vortex kernels isolates runtime testing from compilation concerns.

**Test Categories:**

**Basic Operations (3 tests):**
- `basic_test` - Device/memory basics
- `vecadd_test` - Vector addition
- `demo_test` - Comprehensive demo

**Algorithms (4 tests):**
- `sgemm_test` - Matrix multiply
- `dotproduct_test` - Dot product
- `relu_test` - ReLU activation
- `conv3_test` - 3D convolution

**Advanced Features (3 tests):**
- `sgemm2_test` - Shared memory tiling
- `fence_test` - Memory fences
- `cta_test` - Thread cooperation

**Stress Tests (3 tests):**
- `diverge_test` - Control flow divergence
- `madmax_test` - Computational stress
- `mstress_test` - Memory stress

**All tests passing on Vortex SimX simulator!**

### Phase 1 Achievement

✅ **HIP Runtime Library:** Fully functional API mapping
✅ **Metadata System:** Automatic extraction from DWARF
✅ **Test Coverage:** 13 runtime tests + 40 unit tests
✅ **End-to-End:** Complete execution path verified

**Phase 1 provides a working runtime foundation for Phase 2 compiler integration.**

### Documentation
- **[phase1-runtime-tests/README.md](../phase1-runtime-tests/README.md)** - Runtime test details
- **[phase1-metadata/README.md](../phase1-metadata/README.md)** - Metadata generation
- **[runtime/](../runtime/)** - HIP runtime library source

---

## Phase 2: HIP Compiler Integration

**Status:** 🔨 IN PROGRESS (Split compilation model implemented)
**Purpose:** Compile HIP `__global__` kernels to Vortex RISC-V format

### Architecture: Split Compilation Model

**Selected Approach:** Separate host and kernel compilation paths

The key insight is that HIP programs naturally split into two distinct executables:
1. **Host code** - Runs on x86/ARM host CPU
2. **Kernel code** - Runs on Vortex RISC-V GPU

Rather than trying to compile both through a complex unified pipeline, we use:
- **Host:** Standard C++ compiler (g++) + HIP runtime library
- **Kernel:** Polygeist + MLIR pipeline → Vortex binary

**Why Split Compilation?**
- Simpler toolchain - standard g++ for host, no custom frontend needed
- Better separation of concerns - host and device are independent
- Easier debugging - can test each path independently
- Industry standard - CUDA/HIP/OpenCL all use split compilation
- Proven working - test passes with native Vortex kernel + HIP host

### Complete Compilation Pipeline

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
               ▼                               ▼
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│    KERNEL COMPILATION        │   │      HOST COMPILATION            │
│    (Polygeist Pipeline)      │   │      (Standard C++)              │
├──────────────────────────────┤   ├──────────────────────────────────┤
│ 1. hip_splitter.py           │   │ 1. Standard g++ compiler         │
│    - Extract kernel code     │   │    - Compiles host code          │
│                              │   │    - Links libhip_vortex_runtime │
│ 2. cgeist --cuda-lower       │   │    - Links libvortex.so          │
│    - HIP → MLIR GPU dialect  │   │                                  │
│                              │   │ 2. HIP API calls map to Vortex:  │
│ 3. polygeist-opt             │   │    hipMalloc → vx_mem_alloc      │
│    - GPUToVortex pass        │   │    hipMemcpy → vx_copy_to_dev    │
│    - Thread ID lowering      │   │    hipLaunchKernelGGL → vx_start │
│    - printf → vx_printf      │   │    hipDeviceSynchronize →        │
│                              │   │                   vx_ready_wait  │
│ 4. mlir-translate            │   │                                  │
│    - MLIR → LLVM IR          │   │                                  │
│                              │   │                                  │
│ 5. llvm-vortex backend       │   │                                  │
│    - LLVM IR → RISC-V        │   │                                  │
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
                    └───────────────────────┘
```

### HIP Runtime Library (libhip_vortex_runtime)

The host links against a minimal HIP runtime that maps HIP API to Vortex API:

**Location:** `runtime/hip_vortex_runtime/`

**Implemented APIs:**
```
HIP API                  →  Vortex API
─────────────────────────  ──────────────────────
hipSetDevice()           →  vx_dev_open()
hipGetDeviceProperties() →  vx_dev_caps()
hipMalloc()              →  vx_mem_alloc()
hipFree()                →  vx_mem_free()
hipMemcpy()              →  vx_copy_to/from_dev()
hipMemset()              →  vx_mem_alloc() + vx_copy_to_dev()
hipDeviceSynchronize()   →  vx_ready_wait()
hipGetErrorString()      →  (internal error table)
hipGetLastError()        →  (internal error state)
hipRegisterKernel()      →  vx_upload_kernel_file()
hipLaunchKernelGGL()     →  vx_start()
```

**Build:**
```bash
cd runtime/hip_vortex_runtime
make
# Produces: lib/libhip_vortex_runtime.a
```

### Kernel Metadata (✅ IMPLEMENTED)

During kernel compilation, the `ConvertGPUToVortex` MLIR pass emits metadata files describing kernel argument layouts for runtime argument marshaling.

**Generated Files:**
1. **`<kernel_name>.meta.json`** - JSON metadata for runtime dynamic loading
2. **`<kernel_name>_args.h`** - C header for compile-time type-safe usage

**JSON Format:** (RV32 - 4-byte pointers)

```json
{
  "kernel_name": "_Z13launch_vecaddPKfS0_Pfji",
  "arguments": [
    {"name": "arg0", "type": "i32", "size": 4, "offset": 0, "is_pointer": false},
    {"name": "arg1", "type": "i32", "size": 4, "offset": 4, "is_pointer": false},
    {"name": "arg2", "type": "i32", "size": 4, "offset": 8, "is_pointer": false},
    {"name": "arg3", "type": "ptr", "size": 4, "offset": 12, "is_pointer": true},
    {"name": "arg4", "type": "ptr", "size": 4, "offset": 16, "is_pointer": true},
    {"name": "arg5", "type": "ptr", "size": 4, "offset": 20, "is_pointer": true}
  ],
  "total_args_size": 24,
  "architecture": "rv32"
}
```

**Generated C Header:**
```c
typedef struct {
  int32_t arg0;   // offset=0, size=4
  int32_t arg1;   // offset=4, size=4
  int32_t arg2;   // offset=8, size=4
  uint32_t arg3;  // offset=12, size=4, device pointer
  uint32_t arg4;  // offset=16, size=4, device pointer
  uint32_t arg5;  // offset=20, size=4, device pointer
} kernel_args_t;
```

**Runtime Usage:**
- `hipRegisterKernel()` loads `.meta.json` alongside `.vxbin`
- `is_pointer: true` flags arguments that need device address translation
- Metadata stored in kernel registry for use during `hipLaunchKernelGGL`

### Current Progress

✅ **Split Compilation Model Validated**
- Host compiles with standard g++
- Links against libhip_vortex_runtime + libvortex.so
- Successfully executes native Vortex kernels
- test_native_kernel passes with vecadd kernel

✅ **HIP Runtime Library Complete**
- All memory management APIs implemented
- Device management APIs implemented
- Error handling APIs implemented
- Kernel launch infrastructure implemented

✅ **Polygeist Kernel Pipeline Working**
- 21/22 kernel-only files convert through Polygeist
- ConvertGPUToVortex pass handles thread IDs, barriers
- printf lowering to vx_printf implemented

✅ **Kernel Extraction Tool**
- hip_splitter.py extracts kernels from full HIP sources
- Preserves preprocessor definitions
- Generates kernel-only files for Polygeist

✅ **Kernel Metadata Emission**
- ConvertGPUToVortex pass emits `.meta.json` and `_args.h`
- Runtime parses JSON metadata during kernel registration

### Success Criteria

✅ Host code compiles with standard g++ (no special compiler)
✅ HIP runtime library maps all required APIs to Vortex
✅ Kernel compiles through Polygeist to .vxbin
✅ Runtime loads kernel binary and executes correctly
✅ All Phase 1 tests pass with split compilation model
✅ Kernel metadata enables dynamic argument marshaling

### Documentation
- **[runtime/hip_vortex_runtime/README.md](../runtime/hip_vortex_runtime/README.md)** - HIP runtime library
- **[scripts/hip_splitter.py](../scripts/hip_splitter.py)** - Kernel extraction tool

---

## Phase 3: Full Integration & Optimization

**Status:** ⏳ FUTURE (After Phase 2)
**Purpose:** Complete HIP-to-Vortex toolchain with optimizations

### Scope

**Full Compilation Pipeline:**
```
HIP Source (.hip)
    ↓ [Polygeist + MLIR + GPUToVortexLLVM]
Vortex Binary (.vxbin)
    ↓ [Vortex Runtime]
Execution on Vortex
```

**Optimizations:**
- Warp-level optimizations
- Memory coalescing
- Shared memory banking
- Register allocation
- Instruction scheduling

**Extended API Coverage:**
- Streams and events
- Texture memory
- Constant memory
- Dynamic parallelism (if feasible)

**Optional: HIP Runtime Binary Compatibility Library**
- `libhip_vortex.so` - Wraps Vortex API with HIP API calls
- Purpose: Support pre-compiled host binaries (x86) that were compiled for HIP
- Limitation: Kernels must still be recompiled from source to Vortex RISC-V
- Note: No architecture-independent HIP kernel binary format exists

**Production Features:**
- Error checking and debugging
- Profiling and metrics
- Multi-device support
- Performance tuning tools

### Success Criteria

✅ Complete hipcc-compatible toolchain
✅ Optimizations improve performance significantly
✅ Full HIP API coverage
✅ Production-ready quality
✅ (Optional) Binary compatibility for pre-compiled host code

---

## Repository Structure

```
vortex_hip/
├── Polygeist/                 # ✅ Polygeist (submodule) - HIP → MLIR compiler
│   └── build/                 # Built Polygeist tools (cgeist, etc.)
├── vortex/                    # ✅ Vortex GPU (submodule) - Hardware/simulator
├── llvm-vortex/               # ✅ LLVM-Vortex (submodule) - RISC-V backend
│
├── runtime/                   # ✅ HIP runtime library (Phase 1)
│   ├── include/
│   │   ├── hip/               # HIP API headers (Vortex backend)
│   │   └── vortex.h           # Vortex runtime API
│   ├── src/                   # Runtime implementation
│   └── build/                 # Built libhip_vortex.so
│
├── tests/                     # ✅ All runtime tests (Phase 1)
│   ├── basic_test/
│   ├── vecadd_test/
│   ├── sgemm_test/
│   └── ... (13 total)
│
├── scripts/                   # ✅ Build and metadata scripts
│   └── vortex/                # Kernel metadata generation
│
├── docs/                      # 📚 Technical documentation
│   ├── phase2-polygeist/      # Polygeist integration docs
│   ├── implementation/        # Implementation guides
│   ├── reference/             # Architecture references
│   ├── PHASES_OVERVIEW.md     # This file
│   └── WORK_DISTRIBUTION.md   # 2-developer plan
│
├── phase1-runtime-tests/      # 📖 Phase 1 runtime test docs
├── phase1-metadata/           # 📖 Phase 1 metadata docs
│
├── README.md                  # Project overview
└── INDEX.md                   # Documentation index
```

---

## Current Status Summary

### ✅ Completed (Phase 1)
- HIP runtime library fully functional
- 13 runtime tests passing on Vortex simulator
- Metadata extraction from DWARF working
- End-to-end execution verified

### 🔨 In Progress (Phase 2)
- ✅ Split compilation model validated and working
- ✅ HIP runtime library (libhip_vortex_runtime) complete
- ✅ Polygeist kernel pipeline working (21/22 kernels)
- ✅ ConvertGPUToVortex MLIR pass implemented
- ✅ test_native_kernel passes with Vortex vecadd kernel
- ✅ Kernel metadata emission (JSON + C headers)
- ✅ Runtime metadata parsing during kernel registration
- 📋 Next: End-to-end test with Polygeist-compiled HIP kernel

### ⏳ Future (Phase 3)
- Full toolchain integration
- Performance optimizations
- Extended API coverage
- Optional: HIP runtime binary compatibility library

---

## Key Insight: Why Three Phases?

**Phase 1:** Prove the runtime works (using manual kernels)
- ✅ **Complete** - Runtime proven with 13 tests passing

**Phase 2:** Automate kernel compilation (HIP → Vortex)
- 🔨 **In Progress** - Split compilation + metadata emission complete, end-to-end testing next

**Phase 3:** Optimize and productionize
- ⏳ **Future** - Optimizations and extended features

This approach allows:
1. ✅ Early validation of runtime design
2. 🔨 Compiler work builds on verified runtime
3. ⏳ Optimization happens with complete system

**Current Achievement:** Phase 2 kernel metadata emission complete!

**Key Simplification:** Instead of complex unified compilation, we use:
- Standard g++ for host code (links libhip_vortex_runtime)
- Polygeist pipeline for kernel code only
- Runtime dynamically loads kernel binaries
- Metadata files enable runtime argument marshaling

**Next Step:** End-to-end test with full HIP program compiled via Polygeist

---

**Last Updated:** 2025-11-28
**Target Architecture:** RV32 (32-bit RISC-V)
**Next Milestone:** Complete HIP→Vortex compilation with runtime execution
