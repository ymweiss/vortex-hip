# Vortex HIP Long-Term Integration Plan

**Goal:** Integrate HIP support into the main Vortex repository with proper build system integration.

---

## Current State

```
vortex_hip/                    # Development workspace (NOT the final structure)
├── vortex/                    # Main Vortex repo (submodule)
├── llvm-vortex/               # LLVM 18 + Vortex RISC-V backend (standalone)
├── Polygeist/                 # HIP→MLIR→Vortex compiler
│   └── llvm-project/          # LLVM 18 fork with KernelOutlining mods (submodule)
├── runtime/                   # HIP runtime for Vortex
├── hip_tests/                 # HIP test suite (23 tests, 100% pass)
└── scripts/                   # Build scripts
```

**Issues:**
1. Two separate LLVM builds (llvm-vortex + Polygeist/llvm-project)
2. HIP components scattered across multiple directories
3. No integration with Vortex CI/build system
4. 32-bit only (hardcoded i32 in passes and runtime)

---

## Target State

```
vortex/                        # Primary repository
├── third_party/
│   ├── llvm-vortex/           # Unified LLVM (submodule) - includes KernelOutlining mods
│   ├── polygeist/             # Polygeist passes only (submodule) - uses llvm-vortex
│   └── ... existing ...
├── runtime/
│   ├── simx/                  # Existing SimX runtime
│   ├── rtlsim/                # Existing RTL sim runtime
│   └── hip/                   # NEW: HIP runtime
├── kernel/                    # Existing kernel code
├── tests/
│   ├── opencl/                # Existing OpenCL tests
│   └── hip/                   # NEW: HIP tests
├── ci/
│   ├── toolchain_install.sh   # UPDATED: Optional Polygeist/HIP toolchain
│   └── ...
└── Makefile / CMake           # UPDATED: HIP build targets
```

---

## Integration Phases

### Phase 0: Code Cleanup (Pre-Integration)

**Objective:** Remove deprecated code and clean up documentation before integration work

**Deprecated Passes to Remove:**

| Pass | File | Reason |
|------|------|--------|
| InsertVortexDivergence | `InsertVortexDivergence.cpp` | Branch divergence handled at LLVM backend level (`--vortex-branch-divergence=1` flag to llc) |

**Cleanup Steps:**
1. Remove `InsertVortexDivergence.cpp` from Polygeist passes
2. Remove pass registration from `Passes.h` and `Passes.td`
3. Update CMakeLists.txt to exclude the file
4. Remove any references in documentation

**Documentation Cleanup:**
1. Remove or archive obsolete planning documents (e.g., superseded proposals)
2. Consolidate related documentation into fewer files
3. Update README with current build instructions
4. Remove outdated comments in code referring to old approaches

**Obsolete Documentation to Review:**

| Document | Action |
|----------|--------|
| `LLVM_PROJECT_CHANGES_EVALUATION.md` | Archive - superseded by cherry-pick approach |
| `MAINLINE_INTEGRATION_PLAN.md` | Merge relevant content into this file |
| `REIMPLEMENTATION_PLAN.md` | Archive if no longer relevant |
| `SYNTHETIC_ARG_ELIMINATION_PROPOSAL.md` | Archive if implemented or abandoned |
| `WRAPPER_GENERATION_INVESTIGATION.md` | Keep - still referenced |

**Code Quality:**
1. Remove unused `#include` directives
2. Remove dead code paths
3. Clean up debug `llvm::outs()` statements (keep only warnings/errors)

---

### Phase 1: Consolidate LLVM Submodules

**Objective:** Replace Polygeist's llvm-project with llvm-vortex

**Steps:**
1. Cherry-pick 5 KernelOutlining commits to llvm-vortex branch
2. Update Polygeist to use llvm-vortex as submodule instead of llvm-project
3. Verify Polygeist builds against llvm-vortex
4. Test all 23 HIP tests pass

**Result:** Single LLVM build, ~50% build time reduction

---

### Phase 2: 64-bit Vortex HIP Support

**Objective:** Extend HIP support to RV64 (currently RV32 only)

**Current 32-bit Assumptions:**

| File | Issue |
|------|-------|
| `GenerateVortexMain.cpp:133` | `i32Type` for vx_spawn_threads signature |
| `GenerateVortexMain.cpp:219-220` | Grid/block dim assumed 4 bytes each |
| `ConvertGPUToVortex.cpp` | Multiple `getI32Type()` calls |
| `ConvertGPULaunchToHostCall.cpp:35` | `i32Type` for arguments |
| HIP runtime | Pointer size assumptions in arg marshalling |

**Approach Options:**

| Option | Description | Complexity |
|--------|-------------|------------|
| A: Compile-time | `#ifdef XLEN64` / CMake flag | Low |
| B: Runtime detection | Query data layout from MLIR module | Medium |
| C: Dual ABI | Generate both 32/64 variants | High |

**Clarifications (from discussion):**
- Vortex simulators already support RV64
- Only pointer sizes and data type sizes change (no CSR or ABI changes)
- Host-side runtime requires NO changes (only device code generation)

**Scope:** Parameterize Polygeist passes to use correct pointer width based on target triple.

---

### Phase 3: Vortex Build System Integration

**Objective:** Add HIP as optional build target in Vortex

**Build Script Changes:**

```bash
# ci/toolchain_install.sh additions
hip_toolchain() {
    # Install llvm-vortex with MLIR
    # Install Polygeist (cgeist, polygeist-opt)
    # Install HIP runtime
}

# Detection in build scripts
if [ "$VORTEX_HIP" = "1" ]; then
    # Use HIP compilation path
    compile_hip_kernel "$INPUT" "$OUTPUT"
else
    # Use standard OpenCL/C path
    compile_standard_kernel "$INPUT" "$OUTPUT"
fi
```

**CMake Integration:**
```cmake
option(VORTEX_ENABLE_HIP "Enable HIP/CUDA support" OFF)

if(VORTEX_ENABLE_HIP)
    find_package(Polygeist REQUIRED)
    add_subdirectory(runtime/hip)
    add_subdirectory(tests/hip)
endif()
```

**Questions:**
1. Should HIP be enabled by default or opt-in?
2. Should the HIP toolchain be prebuilt (like riscv-gnu-toolchain) or built from source?
3. What CI systems does Vortex use (GitHub Actions, Jenkins, etc.)?

---

### Phase 4: Repository Restructuring

**Objective:** Move HIP components into Vortex repo structure

**Migration Steps:**
1. Create `third_party/polygeist` as submodule (stripped Polygeist, no llvm-project)
2. Create `third_party/llvm-vortex` as submodule (or reference existing)
3. Move `runtime/hip_vortex_runtime` → `vortex/runtime/hip`
4. Move `hip_tests/` → `vortex/tests/hip`
5. Update CI scripts for HIP toolchain installation
6. Update regression tests to include HIP tests

**Questions:**
1. Should llvm-vortex be a submodule of vortex, or external dependency?
2. Should Polygeist be a submodule or external tool (like clang)?
3. What's the preferred approach for third-party dependencies in Vortex?

---

## Implementation Order

Based on dependencies and risk:

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Code Cleanup (Pre-Integration)                     │
│ - Remove InsertVortexDivergence pass (unused)               │
│ - Archive/remove obsolete documentation                     │
│ - Clean up dead code and debug statements                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: 64-bit Support (FIRST - before submodule changes)  │
│ - Add pointerWidth option to Polygeist passes               │
│ - Parameterize CSR reads, pointer loads, arg offsets        │
│ - Test on RV64 Vortex simulator                             │
│ - See: 64BIT_PARAMETERIZATION_ANALYSIS.md                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Consolidate LLVM Submodules                        │
│ - Cherry-pick KernelOutlining to llvm-vortex                │
│ - Polygeist uses llvm-vortex submodule                      │
│ - Verify 23 HIP tests pass                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Build System Integration                           │
│ - Add HIP toolchain to CI scripts                           │
│ - CMake/Makefile HIP targets                                │
│ - Build detection (HIP vs standard)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Repository Restructuring                           │
│ - Polygeist as submodule                                    │
│ - HIP runtime in vortex/runtime/hip                         │
│ - HIP tests in vortex/tests/hip                             │
│ - Update documentation                                      │
└─────────────────────────────────────────────────────────────┘
```

**Note:** Phase 2 (64-bit) is done BEFORE Phase 1 (submodule consolidation) so all llvm-project changes can be applied at once during cherry-picking.

---

## Questions for Clarification

### Submodule Strategy
1. **llvm-vortex location:** Should it be a submodule of vortex, or remain external?
2. **Polygeist location:** Submodule of vortex, or external dependency like clang?
3. **Polygeist scope:** Full Polygeist or stripped-down (just Vortex passes)?

### 64-bit Support
4. ~~**Priority:** Should 64-bit be done before or after build integration?~~ → **Answered: May be easier first**
5. ~~**RV64 testing:** Is there a working RV64 Vortex simulator?~~ → **Answered: Yes, simulators support RV64**
6. ~~**ABI differences:** Any Vortex-specific ABI changes for RV64?~~ → **Answered: No, only pointer/data sizes change**

### Build System
7. **CI platform:** GitHub Actions, Jenkins, or other?
8. **Prebuilt option:** Should HIP toolchain have prebuilt binaries like riscv-gnu-toolchain?
9. **Default state:** HIP enabled by default or opt-in?

### Relationship
10. **vortex_hip vs vortex:** Is vortex_hip a development fork, or separate project?
11. **Upstream target:** Which branch/repo should PRs target?

---

## Success Criteria

| Phase | Criteria |
|-------|----------|
| 1 | Single LLVM build, 23/23 tests pass |
| 2 | Tests pass on both RV32 and RV64 Vortex |
| 3 | `make hip-tests` works in Vortex repo |
| 4 | HIP fully integrated, CI green |

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| 0: Code cleanup | 0.5-1 day | None |
| 2: 64-bit support | 3-5 days | Phase 0 |
| 1: LLVM consolidation | 2-3 days | Phase 2 |
| 3: Build integration | 2-3 days | Phases 1-2 |
| 4: Repo restructuring | 2-3 days | Phases 1-3 |

**Total:** ~11-15 days
