# Phase 3 Extended Testing Plan

**Date:** 2025-11-07
**Status:** Planning
**Goal:** Validate HIP runtime with existing Vortex test suite

---

## Overview

The `/home/yaakov/vortex_hip/tests/` directory contains 20+ existing HIP tests originally written for Vortex. This document outlines a plan to adapt and validate these tests with our Phase 3 runtime.

## Test Categories

### ✅ Currently Working

**tests/vecadd_metadata_test/** - Vector addition with metadata
- Status: ✅ PASSING (16 elements verified)
- Features: Basic kernel launch, memory operations, metadata marshaling
- Runtime: ~1-2 seconds
- Note: Custom test built specifically for metadata workflow

### 🔍 Ready to Adapt (Simple Tests)

These tests have simple kernels and standard argument patterns that should work with our current runtime:

1. **basic.cpp** - Memory copy and basic kernel
   - Kernel: `__global__ void basic_kernel(int32_t* src, int32_t* dst, uint32_t count)`
   - Features: Memory operations, simple kernel launch
   - Complexity: LOW
   - Priority: HIGH (validates basic functionality)

2. **vecadd.cpp** - Vector addition (production version)
   - Kernel: `__global__ void vecadd_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points)`
   - Features: Templated types, command-line args, verification
   - Complexity: LOW
   - Priority: HIGH (standard benchmark)

3. **fence.cpp** - Memory fence operations
   - Features: Memory synchronization primitives
   - Complexity: LOW
   - Priority: MEDIUM

4. **printf.cpp** - Device-side printf
   - Features: Device printf (may require runtime support)
   - Complexity: MEDIUM
   - Priority: LOW (printf not critical for initial validation)

###

 🚧 Requires Shared Memory (Phase 3 Extension)

These tests use `extern __shared__` which requires runtime support:

5. **dotproduct.cpp** - Dot product with reduction
   - Kernel: `__global__ void dotproduct_kernel(TYPE* src0, TYPE* src1, TYPE* dst, uint32_t num_points)`
   - Features: `extern __shared__ TYPE cache[]`, `__syncthreads()`, reduction
   - Complexity: MEDIUM
   - Priority: HIGH (tests shared memory + synchronization)
   - Note: Requires dynamic shared memory allocation in runtime

6. **sgemm2.cpp** - Matrix multiply with tiling
   - Features: Shared memory tiling for optimization
   - Complexity: MEDIUM-HIGH
   - Priority: MEDIUM

### 🎯 Standard Matrix Operations

7. **sgemm.cpp** - Basic matrix multiplication
   - Kernel: `__global__ void sgemm_kernel(TYPE* A, TYPE* B, TYPE* C, uint32_t size)`
   - Features: 2D grid launch, matrix operations
   - Complexity: MEDIUM
   - Priority: HIGH (standard benchmark)

8. **sgemv.cpp** - Matrix-vector multiply
   - Features: 1D reduction pattern
   - Complexity: MEDIUM
   - Priority: MEDIUM

### 🧠 Neural Network Operations

9. **relu.cpp** - ReLU activation
   - Features: Element-wise operations
   - Complexity: LOW
   - Priority: MEDIUM

10. **dropout.cpp** - Dropout layer
    - Features: Random number generation, element-wise
    - Complexity: MEDIUM
    - Priority: LOW

11. **conv3.cpp** - 3D convolution
    - Features: Complex memory access patterns
    - Complexity: HIGH
    - Priority: LOW

### 🔬 Advanced Tests

12. **diverge.cpp** - Thread divergence
    - Features: Control flow, divergent warps
    - Complexity: MEDIUM
    - Priority: MEDIUM (tests compiler/runtime control flow)

13. **cta.cpp** - Cooperative Thread Arrays
    - Features: CTA operations, advanced synchronization
    - Complexity: HIGH
    - Priority: LOW (advanced feature)

14. **sort.cpp** - Parallel sorting
    - Features: Complex synchronization
    - Complexity: HIGH
    - Priority: LOW

15. **stencil3d.cpp** - 3D stencil
    - Features: Complex memory patterns, halos
    - Complexity: HIGH
    - Priority: LOW

### 🏗️ Infrastructure Tests

16. **madmax.cpp** - Memory access patterns
    - Features: Various access patterns
    - Complexity: MEDIUM
    - Priority: MEDIUM

17. **mstress.cpp** - Memory stress test
    - Features: Large allocations, stress testing
    - Complexity: MEDIUM
    - Priority: MEDIUM

18. **io_addr.cpp** - I/O and addressing
    - Features: Address space handling
    - Complexity: MEDIUM
    - Priority: LOW

### 🎪 Comprehensive Tests

19. **demo.cpp** - Multi-feature demonstration
    - Features: Multiple kernels, various patterns
    - Complexity: MEDIUM-HIGH
    - Priority: MEDIUM

20. **dogfood.cpp** - Comprehensive functionality
    - Features: Everything ("eating our own dog food")
    - Complexity: HIGH
    - Priority: LOW (after all individual tests pass)

### ⚙️ Vortex-Specific

21. **sgemm_tcu.cpp** - Tensor Compute Unit SGEMM
    - Features: Vortex TCU hardware
    - Complexity: VERY HIGH
    - Priority: LOW (requires special hardware)

---

## Adaptation Strategy

### Phase 3A: Simple Tests (Current Focus)

**Goal:** Validate core runtime functionality with simple tests

**Tests:** basic.cpp, vecadd.cpp, sgemm.cpp, relu.cpp

**Approach:**
1. Create Vortex kernel versions (device code)
2. Use existing host code (with metadata integration)
3. Generate metadata using Phase 1 Python script
4. Build with our Makefile pattern from vecadd_metadata_test
5. Run on Vortex simx simulator

**Timeline:** 1-2 days

### Phase 3B: Shared Memory Tests

**Goal:** Add shared memory support to runtime

**Tests:** dotproduct.cpp, sgemm2.cpp

**Requirements:**
- Runtime support for dynamic shared memory (`sharedMemBytes` parameter)
- Proper allocation in argument marshaling
- `__syncthreads()` already supported by Vortex

**Approach:**
1. Update `hipLaunchKernel()` to handle shared memory parameter
2. Pass shared memory size in argument buffer structure
3. Verify with dotproduct test (reduction pattern)

**Timeline:** 2-3 days

### Phase 3C: Advanced Tests

**Goal:** Validate advanced features and edge cases

**Tests:** diverge.cpp, madmax.cpp, mstress.cpp, demo.cpp

**Approach:**
1. Run tests one by one
2. Document any runtime issues
3. Fix issues if simple, defer complex ones to Phase 2

**Timeline:** 3-5 days

### Phase 3D: Neural Network Tests (Optional)

**Tests:** conv3.cpp, dropout.cpp

**Note:** These may require additional library support (random numbers, etc.)

**Timeline:** TBD

---

## Current Limitations (To Be Addressed)

### Known Issues

1. **No dynamic shared memory support**
   - Impact: Tests using `extern __shared__` will fail
   - Solution: Add shared memory handling to runtime

2. **Device printf not implemented**
   - Impact: printf.cpp will not work
   - Solution: Can defer to Phase 2 (not critical)

3. **Standard HIP headers**
   - Impact: Tests expect `<hip/hip_runtime.h>`
   - Solution: Need to provide compatible headers or adapter

4. **Metadata generation required**
   - Impact: All tests need DWARF-based metadata
   - Solution: Use Phase 1 script (temporary), Phase 2 will automate

5. **No texture/surface support**
   - Impact: Tests using textures will fail
   - Solution: Out of scope for Phase 3

### Deferred to Phase 2

The following complexities will be deferred to Phase 2 (compiler integration):

- **Automatic metadata generation** (currently manual DWARF parsing)
- **Complex type handling** (structs, arrays as values)
- **Template instantiation metadata**
- **Device-side printf support**
- **Cooperative groups API**

---

## Test Execution Plan

### Week 1: Simple Kernels

**Day 1-2: Adapt basic.cpp and vecadd.cpp**
- Create kernel.cpp (Vortex device code)
- Use existing main.cpp with minor modifications
- Generate metadata
- Build and test

**Day 3: Adapt sgemm.cpp**
- Matrix multiplication benchmark
- 2D grid launch pattern
- Verify correctness

**Day 4: Adapt relu.cpp**
- Simple element-wise operation
- Different argument pattern

**Day 5: Documentation and analysis**
- Document results
- Create compatibility matrix
- Identify patterns

### Week 2: Shared Memory + Advanced

**Day 1-2: Add shared memory support**
- Update runtime `hipLaunchKernel()`
- Handle `sharedMemBytes` parameter
- Test with simple shared memory kernel

**Day 3: Adapt dotproduct.cpp**
- Reduction with shared memory
- Validate `__syncthreads()`

**Day 4: Adapt sgemm2.cpp**
- Tiled matrix multiplication
- Shared memory optimization

**Day 5: Advanced tests**
- diverge.cpp (control flow)
- madmax.cpp (memory patterns)

### Week 3: Validation + Documentation

**Day 1-3: Run remaining tests**
- Test all adapted kernels
- Document failures and issues
- Create compatibility report

**Day 4-5: Documentation**
- Update Phase 3 README
- Create test results summary
- Plan Phase 2 requirements based on findings

---

## Success Criteria

### Minimum (Phase 3A)

- ✅ vecadd_metadata_test passing (already done)
- ✅ basic.cpp adapted and passing
- ✅ vecadd.cpp adapted and passing
- ✅ sgemm.cpp adapted and passing

**Result:** Core runtime validated with standard benchmarks

### Target (Phase 3B)

- ✅ All Phase 3A tests
- ✅ Shared memory support added
- ✅ dotproduct.cpp passing
- ✅ sgemm2.cpp passing

**Result:** Runtime supports key optimization patterns

### Stretch (Phase 3C)

- ✅ All Phase 3A-B tests
- ✅ 10+ tests passing from existing suite
- ✅ Comprehensive compatibility matrix
- ✅ Performance benchmarks

**Result:** Production-ready runtime for common HIP patterns

---

## Test Infrastructure

### Required Components

1. **Makefile Template**
   - Based on vecadd_metadata_test/Makefile
   - 6-phase build process
   - Metadata generation integrated

2. **Test Runner Script**
   - Based on vecadd_metadata_test/run.sh
   - Environment setup
   - Pass/fail reporting

3. **Verification Framework**
   - Result checking
   - Performance metrics
   - Automated regression testing

### Build System

Each test will follow this structure:

```
tests/<test_name>/
├── kernel.cpp          # Vortex device code
├── main.cpp            # HIP host code (adapted from original)
├── Makefile            # 6-phase build
├── run.sh              # Test runner
└── README.md           # Test-specific notes
```

---

## Compatibility Matrix (To Be Filled)

| Test | Kernel Args | Shared Mem | Special Features | Status | Notes |
|------|-------------|------------|------------------|--------|-------|
| **Phase 3A: Simple Tests (Complete)** |||||
| vecadd_metadata_test | 3 ptr + 1 uint32 | No | - | ✅ PASS | Reference implementation |
| basic.cpp | 3 ptr + 1 uint32 | No | Multi-block | ✅ PASS | 11/11 tests passing |
| vecadd.cpp | 3 ptr + 1 uint32 | No | Multi-block | ✅ PASS | Tested: 16, 256, 1024 elements |
| sgemm.cpp | 3 ptr + 1 uint32 | No | 2D grid | ✅ PASS | Tested: 8x8, 16x16, 64x64 |
| relu.cpp | 3 ptr + 1 uint32 (dummy) | No | Element-wise | ✅ PASS | Tested: 16, 256, 1024 elements |
| fence.cpp | TBD | No | Memory fences | 📝 TODO | Boilerplate created |
| **Phase 3B: Shared Memory** |||||
| dotproduct.cpp | 3 ptr + 1 uint32 | Yes | Reduction | 📝 TODO | Boilerplate created |
| sgemm2.cpp | 3 ptr + 1 uint32 | Yes | Tiling | 📝 TODO | Boilerplate created |
| **Phase 3C: Advanced** |||||
| diverge.cpp | TBD | No | Control flow | 📝 TODO | Boilerplate created |
| madmax.cpp | TBD | No | Memory patterns | 📝 TODO | Boilerplate created |
| mstress.cpp | TBD | No | Stress test | 📝 TODO | Boilerplate created |
| demo.cpp | TBD | Varies | Multi-feature | 📝 TODO | Boilerplate created |
| **Phase 3D: Neural Network** |||||
| sgemv.cpp | TBD | No | Matrix-vector | 📝 TODO | Boilerplate created |
| dropout.cpp | TBD | No | RNG | 📝 TODO | Boilerplate created |
| conv3.cpp | TBD | No | 3D convolution | 📝 TODO | Boilerplate created |
| **Phase 3E: Comprehensive** |||||
| cta.cpp | TBD | Yes | Cooperative groups | 📝 TODO | Boilerplate created |
| sort.cpp | TBD | Yes | Complex sync | 📝 TODO | Boilerplate created |
| stencil3d.cpp | TBD | No | Halo patterns | 📝 TODO | Boilerplate created |
| dogfood.cpp | TBD | Varies | 24 kernels | 📝 TODO | Boilerplate created, comprehensive |
| **Special/Deferred** |||||
| printf.cpp | TBD | No | Device printf | 📝 TODO | Boilerplate created, needs runtime support |
| io_addr.cpp | TBD | No | Address spaces | 📝 TODO | Boilerplate created |
| sgemm_tcu.cpp | TBD | Yes | TCU hardware | 📝 TODO | Boilerplate created, requires special HW |

Legend:
- ✅ PASS - Test passing
- ❌ FAIL - Test failing
- 🔄 TODO - Ready to adapt
- ⏳ WAIT - Waiting for dependencies
- ⛔ SKIP - Out of scope

---

## Next Steps

1. **Immediate (Today)**
   - Start with basic.cpp adaptation
   - Create test directory structure
   - Build and verify

2. **Short-term (This Week)**
   - Complete Phase 3A (4 simple tests)
   - Document results
   - Plan Phase 3B

3. **Medium-term (Next Week)**
   - Implement shared memory support
   - Complete Phase 3B
   - Begin Phase 3C

4. **Long-term (Next 2-3 Weeks)**
   - Complete compatibility matrix
   - Performance benchmarking
   - Transition to Phase 2 planning

---

**Last Updated:** 2025-11-07
**Status:** Initial plan - ready to begin Phase 3A
