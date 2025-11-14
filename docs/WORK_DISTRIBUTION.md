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
[Custom Pass: GPUToVortexLLVM]
│   ├─ Developer A: Thread Model
│   └─ Developer B: Memory Model
    ↓
MLIR LLVM Dialect (with vx_* calls)
    ↓
[mlir-translate --mlir-to-llvmir]
    ↓
LLVM IR (.ll)
    ↓
[llvm-vortex]
    ↓
Vortex RISC-V Binary (.vxbin)
```

---

## Developer A: Thread Model & Synchronization

**Estimated Time:** 2-3 weeks
**Estimated LOC:** ~250-300 lines + tests

### Responsibilities

#### 1. Thread & Block ID Mapping (~150-200 lines)

**Convert GPU dialect thread operations to Vortex runtime calls:**

```mlir
// GPU Dialect → Vortex LLVM
gpu.thread_id x  →  call @vx_thread_id()
gpu.thread_id y  →  call @vx_thread_id() with offset
gpu.thread_id z  →  call @vx_thread_id() with offset

gpu.block_id x   →  compute from vx_warp_id() and thread counts
gpu.block_id y   →  compute from vx_warp_id() and grid dimensions
gpu.block_id z   →  compute from vx_warp_id() and grid dimensions

gpu.global_id    →  blockId * blockDim + threadId
```

**Vortex API Functions:**
- `vx_thread_id()` - Get thread ID within warp
- `vx_warp_id()` - Get warp ID
- `vx_num_threads()` - Get total thread count
- `vx_num_warps()` - Get total warp count

**Implementation Details:**
- Map 3D GPU grid/block model to Vortex's warp-based model
- Handle dimension calculations (x, y, z)
- Compute global thread IDs from local + block IDs
- Handle grid/block dimension queries

#### 2. Synchronization Primitives (~50-100 lines)

**Convert GPU synchronization to Vortex barriers:**

```mlir
gpu.barrier      →  call @vx_barrier(bar_id, num_threads)
gpu.wait         →  appropriate Vortex wait primitive
```

**Vortex API Functions:**
- `vx_barrier(bar_id, num_threads)` - Thread synchronization
- `vx_tmc_reserve()` - Memory consistency operations
- `vx_tmc_acquire()` - Memory consistency operations

**Implementation Details:**
- Map GPU barrier semantics to Vortex barrier implementation
- Handle memory fence requirements
- Ensure correct synchronization scope (warp vs global)

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

## Developer B: Memory Operations & Launch Infrastructure

**Estimated Time:** 2-3 weeks
**Estimated LOC:** ~250-300 lines + tests

### Responsibilities

#### 1. Memory Operations (~150-200 lines)

**Convert GPU memory operations to Vortex memory API:**

```mlir
// Shared Memory
gpu.alloc (shared)  →  vx_shared_mem_allocation()

// Global Memory Transfers
gpu.memcpy (H→D)    →  vx_copy_to_dev(dest, src, size)
gpu.memcpy (D→H)    →  vx_copy_from_dev(dest, src, size)
gpu.memcpy (D→D)    →  vx_mem_copy(dest, src, size)

// Memory Space Mapping
addrspace(1) (global)  →  Vortex global memory
addrspace(3) (shared)  →  Vortex shared memory
addrspace(5) (local)   →  Vortex local/private memory
```

**Vortex API Functions:**
- `vx_copy_to_dev(dest, src, size)` - Host → Device transfer
- `vx_copy_from_dev(dest, src, size)` - Device → Host transfer
- `vx_mem_alloc(size)` - Device memory allocation
- `vx_mem_free(ptr)` - Device memory deallocation

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
