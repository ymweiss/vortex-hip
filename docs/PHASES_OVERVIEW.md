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

**Status:** 🔨 IN PROGRESS (Polygeist integration complete)
**Purpose:** Compile HIP `__global__` kernels to Vortex RISC-V format

### Architecture: Polygeist + MLIR Pipeline

**Selected Approach:** Use Polygeist (official LLVM tool) for HIP → MLIR SCF conversion

**Why Polygeist?**
- Official LLVM project (actively maintained)
- Built-in CUDA/HIP support via `--cuda-lower` flag
- Generates structured MLIR (SCF dialect)
- Standard MLIR passes handle SCF → GPU conversion
- Reduces custom code from ~1000 lines to ~500 lines

### Complete Compilation Pipeline

```
HIP Source (.hip)
    ↓
[Polygeist: cgeist --cuda-lower]
  - Handles __global__, threadIdx, blockIdx, <<<>>>
  - Converts to MLIR SCF (Structured Control Flow)
    ↓
MLIR SCF Dialect
    ↓
[Standard MLIR: --convert-affine-for-to-gpu]
  - SCF → GPU dialect (no custom work needed!)
    ↓
MLIR GPU Dialect
  - gpu.launch_func, gpu.thread_id, gpu.barrier, etc.
    ↓
[Custom Pass: GPUToVortexLLVM] (~500 lines)
  - Developer A: Thread Model & Kernel Launch
  - Developer B: Memory Operations & Argument Marshaling
  - Generates calls to libvortex.so
    ↓
MLIR LLVM Dialect (with vx_* runtime calls)
    ↓
[mlir-translate --mlir-to-llvmir]
    ↓
LLVM IR (.ll)
    ↓
[llvm-vortex backend]
    ↓
Vortex RISC-V Binary (.vxbin)
```

### HIP API Implementation

HIP API calls are handled via **header files** (standard HIP approach):

```cpp
// runtime/include/hip/hip_runtime.h (our Vortex backend)
static inline hipError_t hipMalloc(void** ptr, size_t size) {
    return vx_mem_alloc(vx_get_device(), size, ptr);
}
```

**Flow:**
1. User includes `<hip/hip_runtime.h>` (our version)
2. C preprocessor inlines HIP API → Vortex API calls
3. Polygeist sees `vx_*` calls as regular C functions
4. No special HIP API handling needed in compiler

### Key Transformations in GPUToVortexLLVM Pass

**Device-Side (Kernel Code):**
```mlir
gpu.thread_id x  →  call @vx_thread_id()
gpu.block_id x   →  compute from vx_warp_id()
gpu.barrier      →  call @vx_barrier(bar_id, num_threads)
```

**Host-Side (Kernel Launch):**
```mlir
gpu.launch_func @kernel blocks(...) threads(...)
    ↓
call @vx_upload_kernel_bytes(device, binary, size)
call @vx_start(device)
call @vx_ready_wait(device, timeout)
```

### Current Progress

✅ **Polygeist Built and Validated**
- Successfully built Polygeist from source
- 202MB binary confirms complete build
- `--cuda-lower` flag available and tested

✅ **Documentation Complete**
- Work distribution plan for 2 developers
- Runtime library architecture clarified
- Implementation guides ready

✅ **Submodules Integrated**
- Polygeist (Phase 2 compiler frontend)
- llvm-vortex (RISC-V backend)
- vortex (GPU hardware/simulator)

### Development Plan (5 weeks, 2 developers)

**Week 1: Setup & Infrastructure**
- Test HIP syntax with Polygeist `--cuda-lower`
- Verify standard MLIR passes work
- Set up GPUToVortexLLVM pass framework

**Weeks 2-3: Parallel Implementation**
- Developer A: Thread Model & Kernel Launch (~250 lines)
- Developer B: Memory Operations & Argument Marshaling (~250 lines)

**Week 4: Integration**
- Combine modules
- Metadata extraction from MLIR
- End-to-end testing

**Week 5: Validation**
- Convert all Phase 1 tests to HIP kernels
- Compare results with Phase 1 baselines
- Bug fixes and optimization

### Success Criteria

✅ Polygeist successfully compiles HIP kernels to MLIR
✅ Standard MLIR passes convert SCF → GPU
✅ GPUToVortexLLVM pass generates Vortex runtime calls
✅ All Phase 1 tests pass with compiled HIP kernels
✅ Performance meets or exceeds Phase 1 baselines

### Documentation
- **[docs/WORK_DISTRIBUTION.md](WORK_DISTRIBUTION.md)** - 2-developer plan
- **[docs/phase2-polygeist/](phase2-polygeist/)** - Polygeist integration details
- **[docs/implementation/HIP-TO-VORTEX-API-MAPPING.md](implementation/HIP-TO-VORTEX-API-MAPPING.md)** - API mappings

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
- ✅ Polygeist built and integrated (202MB binary)
- ✅ Submodules configured (Polygeist, llvm-vortex, vortex)
- ✅ Work distribution plan complete (2 developers, 5 weeks)
- ✅ Architecture finalized (Polygeist + MLIR pipeline)
- 📋 Next: Implement GPUToVortexLLVM pass (~500 lines)

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
- 🔨 **In Progress** - Polygeist infrastructure ready, implementing custom pass

**Phase 3:** Optimize and productionize
- ⏳ **Future** - Optimizations and extended features

This approach allows:
1. ✅ Early validation of runtime design
2. 🔨 Compiler work builds on verified runtime
3. ⏳ Optimization happens with complete system

**Current Achievement:** Phase 1 complete, Phase 2 infrastructure ready!

**Critical Path:** Implement GPUToVortexLLVM pass (Developer A: Thread Model, Developer B: Memory Operations)

---

**Last Updated:** 2025-11-14
**Target Architecture:** RV32 (32-bit RISC-V)
**Next Milestone:** Test HIP syntax with Polygeist --cuda-lower flag
