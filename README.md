# HIP-to-Vortex Compilation Pipeline

**Goal:** Enable HIP (Heterogeneous-compute Interface for Portability) applications to run on Vortex RISC-V GPU hardware through an automated compilation pipeline.

**Status:** Phase 1 (Metadata Generation) complete, Phase 2 (Compiler Integration) in progress using Polygeist

---

## Quick Start

**Current Phase:** Building and validating Polygeist for C++/HIP → SCF conversion

See: **[docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)** for complete current status

---

## Project Overview

### What is This Project?

This project implements a **compiler pipeline** that takes standard HIP/CUDA code and compiles it to run on Vortex RISC-V GPU hardware. The pipeline uses:

1. **Polygeist** - Official LLVM tool for C++ → MLIR SCF conversion
2. **MLIR passes** - Standard MLIR transformations (SCF → GPU dialect)
3. **Custom pass** - GPU → Vortex mapping (to be implemented)
4. **llvm-vortex** - RISC-V code generation

### Architecture

```
HIP/CUDA Source Code (.cpp, .cu)
    ↓
[Polygeist] C++ → MLIR SCF Dialect
    ↓
[MLIR Passes] SCF → GPU Dialect
    ↓
[Custom Pass] GPU → LLVM with Vortex calls
    ↓
[mlir-translate] MLIR → LLVM IR
    ↓
[llvm-vortex] LLVM IR → Vortex RISC-V Assembly
    ↓
Vortex Binary
```

**Key Insight:** No LLVM version conflicts - Polygeist (LLVM 18) and llvm-vortex (LLVM 10) never interact directly. Handoff is via standard LLVM IR.

---

## Current Status

### ✅ Phase 1: Metadata Generation (Complete)

**Achievement:** Automatic extraction of kernel argument metadata from compiled binaries

- Python-based metadata generator using DWARF debug information
- Tested with reference HIP kernels
- Generates correct argument offsets, sizes, and pointer flags
- See: `phase1-metadata/` and `docs/PHASE1_COMPLETE.md`

### 🔄 Phase 2: Compiler Integration (In Progress - Using Polygeist)

**Current Work:** Building and validating Polygeist for HIP → SCF conversion

**Completed:**
- ✅ Polygeist built successfully (202MB binary, LLVM 18, optimized)
- ✅ C++ → SCF validation complete (5/5 tests passed)
- ✅ MLIR GPU infrastructure verified available
- ✅ HIP support investigated (47 source references found, 80% likely works as-is)
- ✅ Architecture validated (no version conflicts)

**Current Documentation:**
- **[docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)** - Complete current status
- **[docs/phase2-polygeist/hip_minimal.h](docs/phase2-polygeist/hip_minimal.h)** - Minimal HIP header for testing
- **[docs/phase2-polygeist/investigation/HIP_SUPPORT_INVESTIGATION.md](docs/phase2-polygeist/investigation/HIP_SUPPORT_INVESTIGATION.md)** - HIP support analysis

**Next Steps:**
- **Phase 2A** (2 hours) - Quick test: HIP kernel with `--cuda-lower` flag
- **Phase 2B** (2-3 weeks) - Implement GPUToVortexLLVM pass (~500 lines)
- **Phase 2C** (1 week) - Integrate metadata extraction with compiler

### ⏳ Phase 3: Runtime Library (Planned)

**Goal:** HIP runtime library mapping HIP API calls to Vortex runtime

- Device management (hipGetDeviceCount, hipSetDevice, etc.)
- Memory management (hipMalloc, hipMemcpy, hipFree)
- Kernel execution (hipLaunchKernel, hipDeviceSynchronize)
- See planning in: `phase3-runtime/`

---

## Repository Structure

```
vortex_hip/
├── README.md                          # This file
├── INDEX.md                           # Navigation guide
├── CONTRIBUTING.md                    # Git and documentation guidelines
│
├── docs/phase2-polygeist/             # Polygeist documentation
│   ├── STATUS.md                      # ⭐ Current status
│   ├── BUILD_OPTIONS.md               # Build configuration analysis
│   ├── hip_minimal.h                  # Minimal HIP header
│   ├── HIP_SUPPORT_INVESTIGATION.md   # HIP support findings
│   └── investigation/                 # Investigation results
│
├── phase1-metadata/                   # Phase 1: Metadata extraction ✅
│   ├── hip_metadata_gen.py            # Python metadata generator
│   ├── test_kernels/                  # Reference test kernels
│   └── results/                       # Validation results
│
├── phase2-compiler/                   # Phase 2: Compiler integration 🔄
│   └── [To be created: GPUToVortexLLVM pass]
│
├── phase3-runtime/                    # Phase 3: Runtime library ⏳
│   └── [Planned implementation]
│
├── docs/
│   ├── PHASES_OVERVIEW.md             # Phase breakdown
│   ├── PHASE1_COMPLETE.md             # Phase 1 summary
│   ├── implementation/                # Implementation guides
│   │   ├── HIP-TO-VORTEX-API-MAPPING.md
│   │   ├── COMPILER_INFRASTRUCTURE.md
│   │   ├── COMPILER_METADATA_GENERATION.md
│   │   └── ...
│   └── reference/                     # Architecture documentation
│       └── VORTEX-ARCHITECTURE.md
│
├── vortex/                            # Vortex GPU (git submodule)
└── llvm-vortex/                       # LLVM with Vortex backend (git submodule)
```

---

## Key Technical Decisions

### Why Polygeist?

**Problem:** Need to convert C++/HIP code to MLIR SCF dialect

**Options Evaluated:**
1. Custom LLVM→MLIR pass (complex, error-prone)
2. Clang plugin (requires extensive Clang knowledge)
3. **Polygeist** ⭐ (official LLVM tool, production quality)

**Decision:** Use Polygeist

**Rationale:**
- Official LLVM project (maintained, production-quality)
- Direct C++ → SCF conversion (skips LLVM IR complexity)
- Proven performance (2.5x speedup over alternatives)
- Standard MLIR infrastructure compatibility
- Minimal custom code needed (only GPU → Vortex pass)

### No LLVM Version Conflicts

**Key Architectural Insight:**

```
Polygeist Ecosystem (LLVM 18)          Vortex Ecosystem (LLVM 10)
        ↓                                       ↓
  C++ → SCF → GPU → LLVM Dialect          LLVM IR → RISC-V
        ↓                                       ↑
        └───────── LLVM IR (.ll) ──────────────┘
                   (version-independent)
```

- Polygeist brings its own LLVM 18 (self-contained)
- llvm-vortex uses LLVM 10 (independent)
- Handoff is standard LLVM IR (version-independent)
- **No conflicts, no compatibility issues**

### No Custom Vortex MLIR Dialect Needed

**Discovery:** Vortex uses inline assembly for runtime operations, not LLVM intrinsics

**Implication:** Can emit standard LLVM function calls instead of custom dialect operations

```mlir
gpu.thread_id x  →  call @vx_thread_id()  (standard LLVM)
gpu.block_id x   →  compute from vx_warp_id()
gpu.barrier      →  call @vx_barrier()
```

**Benefit:** Saves ~2000 lines of custom dialect code

---

## Implementation Phases

### Phase 1: Metadata Generation ✅ COMPLETE

**Duration:** 2 weeks
**Status:** Complete and tested

**Deliverables:**
- Python-based DWARF parser
- Automatic argument offset calculation
- Metadata generation for kernel registration
- Validation with reference kernels

**Key Files:**
- `phase1-metadata/hip_metadata_gen.py`
- `docs/PHASE1_COMPLETE.md`

### Phase 2: Compiler Integration 🔄 IN PROGRESS

**Duration:** 4-5 weeks (estimated)
**Status:** Building Polygeist infrastructure

#### Phase 2A: HIP Syntax Testing (2 hours)
- Test HIP kernel with `--cuda-lower` flag
- Verify HIP built-ins work with Polygeist
- Document findings

#### Phase 2B: GPUToVortexLLVM Pass (2-3 weeks)
- Implement custom MLIR pass (~500 lines)
- Map GPU dialect operations to Vortex runtime calls
- `gpu.thread_id` → `call @vx_thread_id()`
- `gpu.block_id` → compute from warp/thread IDs
- `gpu.barrier` → `call @vx_barrier()`

#### Phase 2C: Metadata Integration (1 week)
- Extract metadata from MLIR function signatures
- Integrate with Phase 1 metadata generator
- Generate kernel registration code

**Key Files:**
- `docs/phase2-polygeist/STATUS.md` - Current status
- `phase2-compiler/` - Implementation directory (to be created)

### Phase 3: Runtime Library ⏳ PLANNED

**Duration:** 3-4 weeks (estimated)
**Status:** Planned, not started

**Scope:**
- Host-side HIP API implementation
- Memory management (malloc, memcpy, free)
- Device management (get device count, set device)
- Kernel execution (launch, synchronize)
- Error handling

**Key Files:**
- `phase3-runtime/` - Implementation directory

---

## Quick Links

### Getting Started
1. **[docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)** - Current work and status
2. **[INDEX.md](INDEX.md)** - Full navigation guide
3. **[docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)** - Detailed phase breakdown

### Implementation Guides
- **[docs/implementation/HIP-TO-VORTEX-API-MAPPING.md](docs/implementation/HIP-TO-VORTEX-API-MAPPING.md)** - API mapping reference
- **[docs/implementation/COMPILER_INFRASTRUCTURE.md](docs/implementation/COMPILER_INFRASTRUCTURE.md)** - Compiler architecture
- **[docs/reference/VORTEX-ARCHITECTURE.md](docs/reference/VORTEX-ARCHITECTURE.md)** - Vortex GPU capabilities

### Phase Documentation
- **[docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)** - Phase 1 completion report
- **[phase1-metadata/](phase1-metadata/)** - Phase 1 implementation
- **[docs/phase2-polygeist/](docs/phase2-polygeist/)** - Phase 2 current work

---

## External Dependencies

### Git Submodules

1. **Vortex** - RISC-V GPU hardware
   - Repository: https://github.com/vortexgpgpu/vortex
   - Branch: master
   - Purpose: Target hardware platform

2. **llvm-vortex** - LLVM with Vortex backend
   - Repository: https://github.com/vortexgpgpu/llvm
   - Version: LLVM 10 (custom fork)
   - Purpose: Final RISC-V code generation

3. **Polygeist** - C++ to MLIR translator
   - Repository: https://github.com/ymweiss/Polygeist (fork)
   - Upstream: https://github.com/llvm/Polygeist (official)
   - Version: LLVM 18
   - Purpose: C++/HIP → SCF conversion

**Setup:**
```bash
git clone --recursive https://github.com/YOUR_USERNAME/vortex_hip.git
# Or if already cloned:
git submodule update --init --recursive
```

---

## Development Workflow

### Current Phase 2 Workflow

1. **Polygeist is built** at `/home/yaakov/vortex_hip/docs/phase2-polygeist/build/bin/cgeist`
2. **Test HIP kernels** using `hip_minimal.h` header
3. **Implement GPUToVortexLLVM pass** in `phase2-compiler/`
4. **Integrate with metadata generation** from Phase 1

### Testing

**Phase 1 Tests:**
```bash
cd phase1-metadata
python hip_metadata_gen.py test_kernels/vecadd.hip
```

**Phase 2 Tests (current):**
```bash
cd Polygeist
./build/bin/cgeist simple.cpp -S -o simple.mlir
# Verify SCF dialect output
```

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Commit message conventions
- Documentation standards
- Code style guidelines

---

## Timeline Summary

| Phase | Duration | Status | Key Deliverable |
|-------|----------|--------|-----------------|
| **Phase 1** | 2 weeks | ✅ Complete | Metadata generator |
| **Phase 2A** | 2 hours | 🔄 In Progress | HIP syntax test |
| **Phase 2B** | 2-3 weeks | ⏳ Pending | GPUToVortexLLVM pass |
| **Phase 2C** | 1 week | ⏳ Pending | Metadata integration |
| **Phase 3** | 3-4 weeks | ⏳ Planned | Runtime library |
| **Total** | ~9-11 weeks | ~20% complete | Working HIP compiler |

---

## Resources

### Documentation
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Polygeist Paper](https://mlir.llvm.org/OpenMeetings/2021-01-14-Polygeist.pdf)

### Related Projects
- [Polygeist GitHub](https://github.com/llvm/Polygeist)
- [Vortex GPU](https://github.com/vortexgpgpu/vortex)
- [chipStar](https://github.com/CHIP-SPV/chipStar) - Alternative HIP implementation

---

## Frequently Asked Questions

### Why not use chipStar?

chipStar uses SPIR-V as intermediate representation, which would require:
1. SPIR-V support in Vortex (major effort)
2. SPIR-V → RISC-V translation layer
3. OpenCL runtime implementation

The Polygeist approach is simpler and more direct.

### Why not modify Clang directly?

Polygeist already provides the C++ → MLIR conversion we need. Building a Clang plugin or modifying Clang would duplicate existing functionality.

### How does this compare to official AMD HIP?

This is a **compiler** for Vortex hardware. AMD's HIP is designed for AMD GPUs. We're creating a compatible compiler that targets Vortex RISC-V instead of AMD GCN/CDNA.

### What HIP features are supported?

**Current target (Phase 2):**
- Basic kernel launches
- Memory management (malloc, memcpy, free)
- Thread indexing (threadIdx, blockIdx, blockDim, gridDim)
- Barriers (__syncthreads)

**Future (Phase 3+):**
- Shared memory
- Atomic operations
- Streams and events
- Multi-device support

---

## Contact and Support

For questions about this project, please:
1. Check existing documentation in `docs/`
2. Review phase-specific documentation
3. See implementation guides in `docs/implementation/`

---

**Last Updated:** November 10, 2025
**Current Phase:** Phase 2 (Compiler Integration using Polygeist)
**Status:** Polygeist built and validated, ready for GPUToVortexLLVM pass implementation
