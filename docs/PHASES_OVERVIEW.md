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

**Status:** 📋 PLANNED (Critical next step)
**Purpose:** Compile HIP `__global__` kernels to Vortex RISC-V format

### The Problem

Phase 1 uses **manually written Vortex kernels**:
```cpp
// Vortex format (Phase 1)
#include <vx_spawn.h>
void kernel_body(Args* __UNIFORM__ args) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ...
}
```

We need to compile **HIP kernels** automatically:
```cpp
// HIP format (target for Phase 2)
__global__ void kernel_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

### Required Transformations

**1. Kernel Signature**
- `__global__` → Vortex entry point
- Function parameters → Packed argument struct
- Add runtime fields (grid_dim, block_dim, shared_mem)

**2. Thread Indexing**
- `threadIdx.x/y/z` → Vortex thread intrinsics
- `blockIdx.x/y/z` → Vortex block intrinsics
- `blockDim.x/y/z` → From argument struct
- `gridDim.x/y/z` → From argument struct

**3. Memory Hierarchy**
- `__shared__` → `__local_mem()` allocations
- `__syncthreads()` → Vortex barrier intrinsics
- Global memory → Direct access (already compatible)

**4. Metadata Generation**
- Extract from HIP source (not DWARF)
- Generate registration code
- Pack argument structure

### Implementation Approaches

**Option 1: Clang Plugin** ⭐ (Recommended)
- **Pros:** No LLVM rebuild, easier development, separate distribution
- **Cons:** Limited to AST transformations
- **Best for:** Source-to-source transformation

**Option 2: LLVM Pass**
- **Pros:** Standard workflow, IR-level transformations
- **Cons:** Requires LLVM integration, harder to maintain
- **Best for:** Optimization and lowering

**Option 3: Combined Approach** (Likely best)
- **Clang Plugin:** HIP → Vortex C++ transformation
- **LLVM Pass:** Vortex-specific optimizations
- **Benefits:** Best of both worlds

### Development Plan

**Phase 2A: Basic Kernel Translation**
1. Parse `__global__` functions
2. Transform to Vortex entry point
3. Convert thread/block indexing
4. Handle simple kernels (no shared memory)

**Phase 2B: Memory & Synchronization**
1. `__shared__` → local memory
2. `__syncthreads()` → barriers
3. Memory fence operations

**Phase 2C: Metadata Integration**
1. Extract kernel arguments from AST
2. Generate metadata structure
3. Create registration code
4. Replace DWARF-based system

**Phase 2D: Validation**
1. Convert existing tests to use HIP kernels
2. Compare results with Phase 1 baselines
3. Add HIP-specific test cases

### Timeline Estimate
- **Phase 2A:** 2-3 weeks (basic transformation)
- **Phase 2B:** 1-2 weeks (memory/sync)
- **Phase 2C:** 1-2 weeks (metadata)
- **Phase 2D:** 1 week (testing)
- **Total:** 5-8 weeks

### Success Criteria

✅ Compile HIP `__global__` kernels to Vortex binary
✅ Automatic metadata generation from HIP source
✅ All Phase 1 tests pass with HIP kernels
✅ New HIP-specific features work (shared memory, etc.)

### Documentation
- **[phase2-compiler/README.md](../phase2-compiler/README.md)** - Detailed planning
- **[docs/implementation/COMPILER_INFRASTRUCTURE.md](implementation/COMPILER_INFRASTRUCTURE.md)** - Technical design
- **[docs/implementation/IMPLEMENTATION_CHECKLIST.md](implementation/IMPLEMENTATION_CHECKLIST.md)** - Task breakdown

---

## Phase 3: Full Integration & Optimization

**Status:** ⏳ FUTURE (After Phase 2)
**Purpose:** Complete HIP-to-Vortex toolchain with optimizations

### Scope

**Full Compilation Pipeline:**
```
HIP Source (.cpp)
    ↓ [Clang Plugin]
Vortex C++ (.cpp)
    ↓ [RISC-V Clang]
Vortex Binary (.vxbin)
    ↓ [HIP Runtime]
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

---

## Repository Structure

```
vortex_hip/
├── phase1-runtime-tests/      # ✅ Phase 1 runtime test docs
│   └── README.md
├── phase1-metadata/           # ✅ Phase 1 metadata docs
│   └── README.md
├── phase2-compiler/           # 📋 Phase 2 compiler docs
│   └── README.md
│
├── vortex/                    # Vortex GPU (submodule)
├── llvm-vortex/               # LLVM for Phase 2 (submodule)
│
├── runtime/                   # ✅ HIP runtime library
│   ├── include/               # Public HIP API
│   ├── src/                   # Implementation
│   └── build/                 # Built library
│
├── tests/                     # ✅ All runtime tests (Phase 1)
│   ├── basic_test/
│   ├── vecadd_test/
│   ├── sgemm_test/
│   └── ... (13 total)
│
├── docs/                      # Technical documentation
│   ├── implementation/        # Phase 2 design docs
│   └── reference/             # Vortex architecture
│
├── PHASES_OVERVIEW.md         # This file
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

### 📋 Next Steps (Phase 2)
- Design LLVM pass / Clang plugin architecture
- Implement HIP → Vortex kernel transformation
- Add metadata generation from HIP source
- Validate with existing test suite

### ⏳ Future (Phase 3)
- Full toolchain integration
- Performance optimizations
- Extended API coverage
- Production deployment

---

## Key Insight: Why Three Phases?

**Phase 1:** Prove the runtime works (using manual kernels)
**Phase 2:** Automate kernel compilation (HIP → Vortex)
**Phase 3:** Optimize and productionize

This approach allows:
1. ✅ Early validation of runtime design
2. 📋 Compiler work builds on verified runtime
3. ⏳ Optimization happens with complete system

**Current Achievement:** Phase 1 complete - runtime proven and tested!

**Critical Path:** Phase 2 compiler integration is the key enabler for full HIP support.

---

**Last Updated:** 2025-11-09
**Target Architecture:** RV32 (32-bit RISC-V)
**Next Milestone:** Begin Phase 2 compiler design
