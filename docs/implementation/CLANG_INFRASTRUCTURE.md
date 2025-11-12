# Clang/LLVM Infrastructure for HIP-to-Vortex Compilation

**Document Version:** 1.0
**Date:** 2025-11-09
**Status:** Phase 1 Complete, Phase 2 Planning

---

## Table of Contents

1. [Overview](#overview)
2. [Current Phase 1 Infrastructure](#current-phase-1-infrastructure)
3. [LLVM/Clang Components](#llvmclang-components)
4. [Compilation Pipeline](#compilation-pipeline)
5. [Where Metadata Should Be Generated](#where-metadata-should-be-generated)
6. [Phase 2 Architecture](#phase-2-architecture)
7. [Implementation Options](#implementation-options)

---

## Overview

This document describes the Clang/LLVM compilation infrastructure used for HIP-to-Vortex kernel compilation and identifies where in the compilation pipeline metadata generation should occur.

**Key Question:** At which stage should kernel metadata be extracted/generated?

**Answer:** Metadata should be generated during **Clang frontend processing** (AST stage), NOT from DWARF debug info as currently done in Phase 1.

---

## Current Phase 1 Infrastructure

### Phase 1 Compilation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Manual Vortex Kernels + DWARF Metadata Extraction │
└─────────────────────────────────────────────────────────────┘

kernel.cpp (Vortex format)
    │
    ├─> [1] RISC-V Compilation
    │        Tool: llvm-vortex/bin/clang++
    │        Flags: -march=rv32imaf -mabi=ilp32f -O3 -g (debug info!)
    │        Output: kernel.elf (RISC-V binary with DWARF)
    │
    ├─> [2] DWARF Metadata Extraction (Python)
    │        Tool: vortex/scripts/hip_metadata_gen.py
    │        Input: kernel.elf (reads DWARF debug sections)
    │        Output: kernel_metadata.cpp (registration code)
    │        ⚠️  BUG: Extracts wrong offsets (doesn't skip runtime fields)
    │
    ├─> [3] Binary Conversion
    │        Tool: vortex/kernel/scripts/vxbin.py
    │        Input: kernel.elf
    │        Output: kernel.vxbin (Vortex binary format)
    │
    └─> [4] Host Compilation & Linking
             Tool: g++
             Inputs: main.cpp + kernel_metadata.cpp + kernel_vxbin.o
             Output: test_executable
```

### Current Tools Used

**Kernel Compilation:**
- **Compiler:** `llvm-vortex/bin/clang++` (Vortex-aware LLVM/Clang)
- **Target:** RISC-V 32-bit (rv32imaf)
- **ABI:** ilp32f (32-bit integer, hardware float)
- **Optimizations:** -O3
- **Debug Info:** -g (required for DWARF parsing)

**Metadata Generation:**
- **Tool:** Python script (`vortex/scripts/hip_metadata_gen.py`)
- **Method:** Parse DWARF debug information from compiled ELF
- **Input:** kernel.elf with debug symbols
- **Output:** C++ code with metadata arrays

**Binary Processing:**
- **Tool:** `vxbin.py` (Python)
- **Input:** ELF file
- **Output:** Vortex binary format (.vxbin)
- **Uses:** `llvm-objcopy` for section manipulation

---

## LLVM/Clang Components

### Available LLVM Infrastructure

**Repository:** https://github.com/vortexgpgpu/llvm
**Status:** Fork of LLVM with Vortex-specific modifications

```
llvm-vortex/              # LLVM monorepo with Vortex support
├── llvm/                 # Core LLVM libraries
│   ├── include/llvm/     # LLVM IR, passes, transforms
│   ├── lib/              # Implementation
│   │   └── Target/RISCV/ # RISC-V backend
│   │       ├── VortexIntrinsicFunc.cpp  ⭐ Vortex intrinsic lowering
│   │       ├── RISCVTargetMachine.cpp   (modified for Vortex)
│   │       └── RISCV.h                  (Vortex extensions)
│   └── tools/            # LLVM utilities
│
├── clang/                # Clang C/C++ frontend
│   ├── include/clang/    # AST, Sema, CodeGen
│   │   ├── AST/          # Abstract Syntax Tree
│   │   ├── Sema/         # Semantic analysis
│   │   ├── CodeGen/      # LLVM IR generation
│   │   └── Frontend/     # Frontend actions, plugins
│   ├── lib/              # Implementation
│   └── tools/            # Clang driver, tools
│
├── compiler-rt/          # Runtime libraries
└── libcxx/               # C++ standard library
```

**Vortex-Specific Modifications:**
- **Vortex Intrinsic Lowering:** `VortexIntrinsicFunc.cpp` - LLVM pass for Vortex intrinsics
- **Divergence Support:** Split/join extensions for control flow divergence
- **Target Features:** `-Xclang -target-feature -Xclang +vortex` flag
- **RISC-V Extensions:** Vortex-specific instruction definitions

### Key Clang Components for Phase 2

**1. Clang AST (Abstract Syntax Tree)**
- Location: `clang/include/clang/AST/`
- Purpose: Represents parsed C++ code structure
- Use: Parse `__global__` functions, extract parameters

**2. Clang Frontend**
- Location: `clang/include/clang/Frontend/`
- Purpose: Provides plugin framework
- Use: Hook into compilation to transform HIP code

**3. Clang Sema (Semantic Analysis)**
- Location: `clang/include/clang/Sema/`
- Purpose: Type checking, name resolution
- Use: Validate HIP constructs, resolve types

**4. Clang CodeGen**
- Location: `clang/include/clang/CodeGen/`
- Purpose: Generate LLVM IR from AST
- Use: Transform HIP kernels to Vortex-compatible IR

**5. LLVM Passes**
- Location: `llvm/include/llvm/Transforms/`
- Purpose: Optimize and transform LLVM IR
- Use: Vortex-specific optimizations

---

## Compilation Pipeline

### Detailed Clang Compilation Stages

```
HIP Source Code (kernel.hip)
    │
    ▼
┌────────────────────────────────────────────┐
│ [1] PREPROCESSING                          │
│     Tool: Clang preprocessor               │
│     Action: Expand macros, includes        │
│     Output: Preprocessed source            │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [2] PARSING (Lexical Analysis + Parsing)   │
│     Tool: Clang parser                     │
│     Action: Create Abstract Syntax Tree    │
│     Output: Clang AST                      │
│                                            │
│     ⭐ METADATA STAGE 1: Parse Kernels     │
│     - Identify __global__ functions        │
│     - Extract function signatures          │
│     - Collect parameter information        │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [3] SEMANTIC ANALYSIS (Sema)               │
│     Tool: Clang semantic analyzer          │
│     Action: Type checking, name resolution │
│     Output: Annotated AST                  │
│                                            │
│     ⭐ METADATA STAGE 2: Type Analysis     │
│     - Resolve parameter types              │
│     - Determine sizes and alignments       │
│     - Identify pointers vs scalars         │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [4] AST TRANSFORMATION                     │
│     Tool: Clang plugin or custom pass      │
│     Action: Transform HIP to Vortex        │
│     Output: Modified AST                   │
│                                            │
│     🔧 PHASE 2 WORK HAPPENS HERE:         │
│     - Transform __global__ → vortex entry  │
│     - Convert threadIdx/blockIdx           │
│     - Handle __shared__ memory             │
│     - Generate argument structure          │
│                                            │
│     ⭐ METADATA STAGE 3: Generate Metadata │
│     - Create metadata arrays               │
│     - Generate registration code           │
│     - Emit as separate compilation unit    │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [5] CODE GENERATION (CodeGen)              │
│     Tool: Clang CodeGen                    │
│     Action: AST → LLVM IR                  │
│     Output: LLVM IR (.ll or .bc)           │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [6] LLVM OPTIMIZATION                      │
│     Tool: LLVM opt                         │
│     Action: Optimize IR                    │
│     Output: Optimized LLVM IR              │
│                                            │
│     🔧 Optional Vortex-specific passes     │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [7] MACHINE CODE GENERATION                │
│     Tool: LLVM backend                     │
│     Action: IR → RISC-V assembly           │
│     Output: kernel.s (assembly)            │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [8] ASSEMBLY & LINKING                     │
│     Tool: RISC-V assembler/linker          │
│     Action: Create binary                  │
│     Output: kernel.elf                     │
└────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────┐
│ [9] BINARY CONVERSION                      │
│     Tool: vxbin.py                         │
│     Action: ELF → Vortex format            │
│     Output: kernel.vxbin                   │
└────────────────────────────────────────────┘
```

---

## Where Metadata Should Be Generated

### ⭐ CORRECT APPROACH: AST-Level Metadata Generation

**Stage:** After Semantic Analysis, during AST Transformation (Stage 4)

**Why this stage?**

1. **Type Information Available**
   - Semantic analysis has resolved all types
   - Sizes, alignments, qualifiers are known
   - Pointer vs scalar distinction is clear

2. **Source-Level Understanding**
   - Can directly see `__global__` function parameters
   - No need to reverse-engineer from DWARF
   - Access to full type hierarchy

3. **Before Code Generation**
   - Can influence code generation
   - Can emit metadata as compilation output
   - Can transform code based on metadata

4. **Clean Separation**
   - Metadata generation separate from runtime fields
   - Direct access to user-defined parameters
   - No confusion with compiler-generated fields

### Implementation Point: Clang Plugin

**Recommended:** Implement as Clang Plugin that runs during compilation

**Plugin Hook:** `ASTConsumer` or `FrontendAction`

**Timing:** After semantic analysis, before CodeGen

**What the Plugin Does:**

```cpp
// Pseudo-code for Clang plugin

class HIPMetadataGenerator : public PluginASTAction {
  void ExecuteAction() override {
    // Get the AST context
    ASTContext &Context = getCompilerInstance().getASTContext();

    // Find all __global__ functions
    for (auto *Decl : Context.getTranslationUnitDecl()->decls()) {
      if (auto *FD = dyn_cast<FunctionDecl>(Decl)) {
        if (FD->hasAttr<CUDAGlobalAttr>()) {  // __global__ attribute

          // Extract parameter metadata
          for (auto *Param : FD->parameters()) {
            QualType Type = Param->getType();

            // Get size, alignment, pointer flag
            uint64_t size = Context.getTypeSize(Type) / 8;
            uint64_t alignment = Context.getTypeAlign(Type) / 8;
            bool is_pointer = Type->isPointerType();

            // Store metadata
            metadata.emplace_back(size, alignment, is_pointer);
          }

          // Generate metadata code
          emitMetadataRegistration(FD->getName(), metadata);
        }
      }
    }
  }
};
```

### ❌ INCORRECT APPROACH: DWARF-Level Metadata (Current Phase 1)

**Stage:** After compilation, from debug information

**Problems:**

1. **Extracts Wrong Information**
   - Sees ALL struct fields, including runtime fields
   - Cannot distinguish user args from runtime args
   - Reports incorrect offsets

2. **Requires Debug Info**
   - Must compile with -g flag
   - Larger binaries
   - Extra compilation step

3. **Fragile**
   - Depends on DWARF format stability
   - Sensitive to compiler optimizations
   - Hard to debug when wrong

4. **No Source Context**
   - Lost connection to original HIP code
   - Cannot validate HIP constructs
   - Cannot perform HIP-specific checks

---

## Phase 2 Architecture

### Recommended Phase 2 Design

```
┌──────────────────────────────────────────────────────────┐
│ Phase 2: HIP Kernel Compilation with AST-Level Metadata  │
└──────────────────────────────────────────────────────────┘

HIP Source (kernel.hip with __global__ functions)
    │
    ├─> [1] Clang Frontend (with HIP plugin)
    │        - Parse HIP source
    │        - Identify __global__ kernels
    │        - Semantic analysis
    │
    ├─> [2] Clang Plugin: HIP→Vortex Transformation
    │        ┌────────────────────────────────────┐
    │        │ A. Transform __global__ function   │
    │        │    - Rename to kernel_body         │
    │        │    - Add vx_spawn_threads wrapper  │
    │        │                                    │
    │        │ B. Transform thread indexing       │
    │        │    - threadIdx.x → Vortex builtin  │
    │        │    - blockIdx.x → Vortex builtin   │
    │        │                                    │
    │        │ C. Transform memory                │
    │        │    - __shared__ → __local_mem()    │
    │        │    - __syncthreads() → barrier     │
    │        │                                    │
    │        │ D. Generate argument structure     │
    │        │    struct KernelArgs {             │
    │        │      uint32_t grid_dim[3];         │
    │        │      uint32_t block_dim[3];        │
    │        │      uint64_t shared_mem;          │
    │        │      // User params from AST       │
    │        │    };                              │
    │        │                                    │
    │        │ ⭐ E. Generate Metadata (AST)      │
    │        │    - Extract user parameters ONLY  │
    │        │    - Compute correct offsets       │
    │        │    - Emit metadata arrays          │
    │        │    - Generate registration code    │
    │        └────────────────────────────────────┘
    │        Output: Transformed AST + metadata_gen.cpp
    │
    ├─> [3] CodeGen: AST → LLVM IR
    │        Output: kernel.ll (LLVM IR)
    │
    ├─> [4] LLVM Optimization
    │        Optional: Vortex-specific passes
    │        Output: Optimized IR
    │
    ├─> [5] RISC-V Code Generation
    │        Target: rv32imaf
    │        Output: kernel.elf
    │
    ├─> [6] Binary Conversion
    │        Tool: vxbin.py
    │        Output: kernel.vxbin
    │
    └─> [7] Host Compilation
             Link: main.cpp + metadata_gen.cpp + kernel_vxbin.o
             Output: Final executable
```

### Key Differences from Phase 1

| Aspect | Phase 1 (Current) | Phase 2 (Target) |
|--------|-------------------|------------------|
| **Kernel Format** | Manual Vortex C++ | HIP `__global__` kernels |
| **Metadata Source** | DWARF debug info | Clang AST |
| **Metadata Timing** | Post-compilation | During compilation |
| **Transformation** | None (manual) | Automatic (Clang plugin) |
| **Correctness** | Bug in offsets | Direct from source |
| **Tooling** | Python script | Clang plugin + LLVM |

---

## Implementation Options

### Option 1: Clang Plugin ⭐ (Recommended)

**Approach:** External Clang plugin loaded at compile time

**Pros:**
- No LLVM source modification required
- Can be distributed separately
- Easier development and testing
- Faster iteration cycle
- Used by other HIP implementations (ROCm)

**Cons:**
- Limited to AST-level transformations
- Cannot modify LLVM backend directly
- Plugin API has some limitations

**Implementation:**
```bash
# Compile plugin
clang++ -shared -fPIC hip_to_vortex_plugin.cpp \
    -I$LLVM_SRC/clang/include -o hip_to_vortex.so

# Use plugin
clang++ -fplugin=./hip_to_vortex.so \
    -target riscv32 kernel.hip -o kernel.elf
```

**Files to Create:**

**Location:** Either in llvm-vortex or vortex_hip repository

**Option A: In llvm-vortex repo (integrated with existing Vortex code):**
- `llvm-vortex/clang/lib/HIPToVortex/`
  - `HIPTransform.cpp` - Main plugin logic
  - `MetadataGenerator.cpp` - Metadata extraction
  - `ThreadIndexRewriter.cpp` - threadIdx/blockIdx transform
  - `SharedMemoryHandler.cpp` - __shared__ transform

**Option B: In vortex_hip repo (separate from LLVM):**
- `vortex_hip/compiler/plugins/hip_to_vortex/`
  - `HIPTransform.cpp` - Main plugin logic
  - `MetadataGenerator.cpp` - Metadata extraction
  - `ThreadIndexRewriter.cpp` - threadIdx/blockIdx transform
  - `SharedMemoryHandler.cpp` - __shared__ transform

**Recommendation:** Option A (in llvm-vortex) to keep compiler infrastructure together with existing Vortex modifications

### Option 2: LLVM Pass

**Approach:** Custom LLVM IR transformation pass

**Pros:**
- Works on LLVM IR (lower level)
- Can perform advanced optimizations
- Standard LLVM workflow

**Cons:**
- Requires LLVM build integration
- Harder to maintain
- Lost source-level information

**When to Use:**
- For Vortex-specific IR optimizations
- After AST transformation
- Complementary to Clang plugin

### Option 3: Combined Approach (Recommended Long-term)

**Clang Plugin:** HIP → Vortex transformation + metadata generation
**LLVM Pass:** Vortex-specific IR optimizations

**Workflow:**
1. Clang plugin transforms HIP source to Vortex C++
2. Clang plugin generates metadata from AST
3. CodeGen produces LLVM IR
4. LLVM pass optimizes for Vortex
5. Backend generates RISC-V code

---

## Summary: Where Metadata Should Be Generated

### ✅ Correct Answer: Clang AST Stage

**Stage:** After semantic analysis, during AST transformation (Stage 4)

**Tool:** Clang Plugin implementing ASTConsumer

**Input:** Clang AST with type information

**Process:**
1. Iterate over `__global__` function declarations
2. Extract parameter list from AST
3. Query type system for sizes, alignments
4. Compute offsets (accounting for runtime fields)
5. Generate metadata arrays
6. Emit registration code

**Output:**
- `kernel_metadata.cpp` with correct offsets
- Embedded in compilation output
- No dependency on DWARF

### ❌ Current Phase 1 Approach (Incorrect)

**Stage:** Post-compilation, from DWARF

**Tool:** Python script parsing debug info

**Problems:**
- Wrong offsets (doesn't skip runtime fields)
- Requires debug symbols
- Fragile, hard to maintain
- Lost source context

### 🎯 Phase 2 Priority #1

**Fix metadata generation by moving it to Clang AST stage**

This is the foundation for all Phase 2 work and must be completed before HIP kernel transformation can work correctly.

---

## Next Steps

1. **Design Clang Plugin Architecture** (Week 1)
   - Define plugin interface
   - Plan AST visitor pattern
   - Design metadata output format

2. **Implement Basic Plugin** (Week 2)
   - Parse `__global__` functions
   - Extract parameter metadata
   - Generate registration code

3. **Validate with Phase 1 Tests** (Week 2)
   - Run all 14 tests with new metadata
   - Verify correct offsets
   - Remove dummy argument workarounds

4. **Add HIP Transformations** (Weeks 3-4)
   - Thread indexing
   - Shared memory
   - Synchronization

5. **Convert Tests to HIP** (Weeks 5-6)
   - Rewrite kernels using `__global__`
   - Validate against Phase 1 baselines

---

**Document Status:** Complete
**Last Updated:** 2025-11-09
**Next Review:** After Phase 2A implementation
