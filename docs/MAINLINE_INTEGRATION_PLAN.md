# Mainline Vortex Integration Plan

This document outlines the plan for integrating HIP-to-Vortex compilation support into mainline Vortex with minimal dependencies.

## Current State

The HIP-to-Vortex pipeline currently requires:

| Component | Size | Purpose |
|-----------|------|---------|
| Polygeist fork | ~7.5GB | Clang → MLIR frontend + Vortex passes |
| llvm-vortex | ~200GB build | Custom RISC-V backend for Vortex ISA |

**Total build-time footprint: ~200GB+**

## Target State

| Component | Size | Source |
|-----------|------|--------|
| Stock LLVM/MLIR | System package | HIP parsing + MLIR infrastructure |
| Vortex MLIR passes | ~50KB source | Extracted to vortex repo |
| llvm-vortex | Prebuilt binaries | vortex-toolchain-prebuilt |
| Device stubs | ~20KB | Extracted to vortex repo |

**Target footprint: System LLVM + prebuilts only**

---

## Component Analysis

### 1. llvm-vortex (No Change Needed)

**Status:** Already part of Vortex ecosystem

llvm-vortex provides the custom RISC-V backend for Vortex ISA extensions. It is already:
- Maintained at https://github.com/vortexgpgpu/llvm-vortex
- Available as prebuilt binaries via vortex-toolchain-prebuilt
- Downloaded by `vortex/ci/toolchain_install.sh.in`

**Action:** No changes needed. Continue using existing infrastructure.

### 2. Polygeist Fork (Primary Extraction Target)

The Polygeist fork serves two purposes:

#### A. cgeist Frontend (HIP → MLIR)
- Parses HIP/CUDA source files
- Generates MLIR with gpu.launch_func operations
- Handles `__global__`, `__device__`, `__shared__` attributes

#### B. Vortex MLIR Passes (gpu.* → Vortex runtime)

| Pass | Purpose |
|------|---------|
| ConvertGPUToVortex | Lowers gpu.launch_func → vx_spawn_threads calls |
| GenerateVortexMain | Creates vortex_kernel entry point for .vxbin |
| InsertVortexDivergence | Adds divergence-safe control flow |
| ConvertGPULaunchToHostCall | Creates launch wrappers for host code |
| ReorderGPUKernelArgs | Reorders kernel arguments for ABI |
| StripHostOnlyFunctions | Removes host functions from device module |
| ConvertParallelToGPU | Converts parallel loops to GPU operations |

### 3. Device Stub Headers

Located in `Polygeist/tools/cgeist/include/polygeist_device_stubs/`:
- Intercept STL headers during CUDA compilation
- Prevent `__float128` errors from GCC's libstdc++
- ~15 header files, ~20KB total

---

## Integration Strategy

### Phase 1: Extract Vortex Passes as Standalone Plugin

**Goal:** Build Vortex MLIR passes against stock LLVM/MLIR

**Steps:**
1. Create `vortex/tools/hip-compiler/` directory structure
2. Extract the 7 Vortex MLIR passes
3. Create CMakeLists.txt that finds system MLIR
4. Build as `libVortexMLIRPasses.so` plugin
5. Load with `mlir-opt --load-pass-plugin`

**Directory structure:**
```
vortex/tools/hip-compiler/
├── CMakeLists.txt
├── lib/
│   └── Passes/
│       ├── ConvertGPUToVortex.cpp
│       ├── GenerateVortexMain.cpp
│       ├── InsertVortexDivergence.cpp
│       └── ...
├── include/
│   └── device_stubs/
│       ├── iostream
│       ├── vector
│       └── ...
└── scripts/
    └── compile_hip.sh
```

**CMake integration:**
```cmake
find_package(MLIR REQUIRED CONFIG)
add_mlir_library(VortexMLIRPasses
  lib/Passes/ConvertGPUToVortex.cpp
  lib/Passes/GenerateVortexMain.cpp
  ...
  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRLLVMDialect
  MLIRTransforms
)
```

### Phase 2: Replace cgeist with Stock Clang

**Goal:** Use Clang's CUDA frontend directly instead of cgeist

**Current pipeline (cgeist-based):**
```
HIP source → cgeist --emit-cuda → MLIR → polygeist-opt → llc → vxbin
```

**Target pipeline (stock Clang-based):**
```
HIP source → clang -emit-llvm → LLVM IR → mlir-translate → MLIR → mlir-opt --vortex-passes → llc → vxbin
```

**Challenges:**
- Clang's CUDA frontend emits LLVM IR, not MLIR
- Need mlir-translate or custom importer for LLVM IR → MLIR GPU dialect
- Alternative: Keep cgeist but build it against stock LLVM

**Recommendation:**
For Phase 2, evaluate whether:
- A. Using `mlir-translate --import-llvm` is sufficient
- B. A lightweight cgeist rebuild against stock LLVM is simpler

### Phase 3: Runtime Header Integration

**Goal:** Integrate HIP runtime headers into Vortex

**Files to migrate:**
```
runtime/include/hip/hip_runtime.h     → vortex/runtime/hip/hip_runtime.h
runtime/device/hip/hip_runtime.h      → vortex/runtime/hip/device/hip_runtime.h
runtime/host/hip/hip_runtime.h        → vortex/runtime/hip/host/hip_runtime.h
```

These headers map HIP API calls to Vortex API calls:
- `hipMalloc` → `vx_mem_alloc`
- `hipMemcpy` → `vx_copy_to_dev` / `vx_copy_from_dev`
- `hipDeviceSynchronize` → `vx_ready_wait`

---

## MLIR Version Compatibility

The Vortex MLIR passes use these dialects:
- `gpu` dialect (kernel launch, thread IDs)
- `llvm` dialect (generated code)
- `func` dialect (function definitions)
- `arith` dialect (arithmetic operations)
- `memref` dialect (memory references)

**Minimum LLVM version:** 17 (for GPU dialect stability)
**Recommended:** LLVM 18+ (better GPU lowering)

---

## Build Requirements After Integration

| Requirement | Package | Notes |
|-------------|---------|-------|
| LLVM/Clang | llvm-18-dev, clang-18 | System packages |
| MLIR | libmlir-18-dev | Usually bundled with LLVM |
| llvm-vortex | Prebuilt | From vortex-toolchain-prebuilt |

**Ubuntu 24.04:**
```bash
apt install llvm-18-dev clang-18 libmlir-18-dev
```

**Build time:** ~5 minutes (passes only, vs hours for full Polygeist)

---

## Migration Checklist

### Vortex Repository Changes

- [ ] Create `vortex/tools/hip-compiler/` directory
- [ ] Port MLIR passes with stock LLVM compatibility
- [ ] Add device stub headers
- [ ] Add HIP runtime headers
- [ ] Add compilation script
- [ ] Update build system (CMake)
- [ ] Add documentation
- [ ] Ensure device/host splitting removes unused kernel artifacts from host binaries

### Testing

- [ ] Verify all 23 HIP tests compile
- [ ] Verify kernels execute correctly on Vortex simulator
- [ ] Test on multiple LLVM versions (17, 18, 19)
- [ ] CI integration

### Documentation

- [ ] Update Vortex docs with HIP compilation instructions
- [ ] Add examples
- [ ] Document LLVM version requirements

---

## Development Constraints

### No LLVM/MLIR Core Modifications

**Constraint:** No changes can be made to LLVM/MLIR core code (`Polygeist/llvm-project/`).

**Rationale:** Reducing storage requirements requires using builtin LLVM instead of a modified fork. Building a custom LLVM fork requires ~200GB of storage and hours of build time. Using system LLVM packages eliminates this requirement.

**Scope:** All modifications must be in Polygeist-specific code:
- `Polygeist/tools/cgeist/` - Clang frontend modifications
- `Polygeist/lib/polygeist/` - Vortex MLIR passes

**Impact on development:**
- Cannot modify `KernelOutlining.cpp` or other LLVM/MLIR passes
- Must work around LLVM limitations through Polygeist-level code
- Solution patterns must be compatible with stock LLVM behavior

**Example workaround:** The kernel argument reordering issue in `KernelOutlining.cpp` cannot be fixed directly. Instead, launch wrappers are generated at the clang AST level (in `clang-mlir.cc`) where arguments can be properly traced.

See `docs/WRAPPER_GENERATION_INVESTIGATION.md` for detailed investigation of wrapper-based solutions.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLIR API changes between LLVM versions | Medium | Pin minimum version, use stable APIs |
| cgeist features hard to replicate | Low | Can build minimal cgeist against stock LLVM |
| GPU dialect changes | Low | Use stable gpu.launch_func patterns |

---

## Timeline Estimate

| Phase | Scope | Dependency |
|-------|-------|------------|
| Phase 1 | Extract passes | None |
| Phase 2 | Stock Clang frontend | Phase 1 |
| Phase 3 | Runtime integration | Phase 1 |
| Testing | Full validation | All phases |

---

## Current Pass Rate

With Polygeist fork: **17/23 tests (74%)** compile successfully.

Remaining failures require:
- Device math functions (min/max/pow)
- Memory fence intrinsics
- Device function handling
- Host `__shared__` macro fixes

These are orthogonal to the integration plan and can be fixed before or after migration.
