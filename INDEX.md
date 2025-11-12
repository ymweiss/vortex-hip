# Vortex HIP Project Index

**Quick navigation to all project documentation and implementation files.**

**Last Updated:** November 10, 2025
**Current Phase:** Phase 2 - Polygeist Integration
**Status:** Polygeist built and validated, implementing compiler passes

---

## 🚀 Quick Start

**Want to understand the current state?**
1. Start with **[README.md](README.md)** - Project overview
2. Check **[docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)** - Current work status
3. Review **[docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)** - Implementation phases

**Want to see what's been completed?**
- **[docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)** - Phase 1 completion report (metadata generation)

---

## 📚 Documentation Structure

### Primary Documents (Start Here)

| Document | Purpose | Status |
|----------|---------|--------|
| [README.md](README.md) | Project overview and getting started | ✅ Current |
| [docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md) | **Current work status** | ✅ Current |
| [docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md) | Implementation phases | ✅ Current |
| [docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md) | Phase 1 summary | ✅ Complete |
| [INDEX.md](INDEX.md) | This navigation guide | ✅ Current |

### Implementation Guides

| Guide | Topic | Location |
|-------|-------|----------|
| HIP to Vortex API Mapping | Complete API reference | [docs/implementation/HIP-TO-VORTEX-API-MAPPING.md](docs/implementation/HIP-TO-VORTEX-API-MAPPING.md) |
| Compiler Infrastructure | Compiler architecture | [docs/implementation/COMPILER_INFRASTRUCTURE.md](docs/implementation/COMPILER_INFRASTRUCTURE.md) |
| Metadata Generation | Metadata extraction details | [docs/implementation/COMPILER_METADATA_GENERATION.md](docs/implementation/COMPILER_METADATA_GENERATION.md) |
| Metadata Summary | Implementation overview | [docs/implementation/METADATA_GENERATION_SUMMARY.md](docs/implementation/METADATA_GENERATION_SUMMARY.md) |
| Implementation Checklist | Progress tracking | [docs/implementation/IMPLEMENTATION_CHECKLIST.md](docs/implementation/IMPLEMENTATION_CHECKLIST.md) |

### Reference Documentation

| Reference | Content | Location |
|-----------|---------|----------|
| Vortex Architecture | GPU capabilities and API | [docs/reference/VORTEX-ARCHITECTURE.md](docs/reference/VORTEX-ARCHITECTURE.md) |

---

## 💻 Implementation by Phase

### Phase 1: Metadata Generation ✅ COMPLETE

**Status:** Complete and tested
**Duration:** 2 weeks
**Completion:** November 2025

**Key Files:**
- [phase1-metadata/](phase1-metadata/) - Implementation directory
- [scripts/vortex/hip_metadata_gen.py](scripts/vortex/hip_metadata_gen.py) - Main generator
- [docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md) - Completion report

**What It Does:**
- Parses DWARF debug information from compiled kernels
- Extracts kernel argument metadata (offset, size, is_pointer)
- Generates kernel registration code
- Validates with reference HIP kernels

---

### Phase 2: Compiler Integration 🔄 IN PROGRESS

**Status:** Building Polygeist infrastructure
**Duration:** 4-5 weeks (estimated)
**Started:** November 2025

**Current Subphase:** Phase 2A - HIP Syntax Testing

#### Polygeist Build ✅ Complete

**Key Directory:** [docs/phase2-polygeist/](docs/phase2-polygeist/)

**Documentation:**
- [docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md) - **Primary reference**
- [docs/phase2-polygeist/hip_minimal.h](docs/phase2-polygeist/hip_minimal.h) - Minimal HIP header for testing
- [docs/phase2-polygeist/investigation/HIP_SUPPORT_INVESTIGATION.md](docs/phase2-polygeist/investigation/HIP_SUPPORT_INVESTIGATION.md) - HIP support analysis

**Build Output:**
- Binary: `docs/phase2-polygeist/build/bin/cgeist` (202MB)
- LLVM Version: 18.0.0git (optimized)

**Validation:** 5/5 tests passed
- Simple C → MLIR ✅
- Loops → SCF dialect ✅
- Nested loops ✅
- Conditionals → SCF/Arith ✅
- CUDA infrastructure present ✅

#### Phase 2A: HIP Testing (Current) 🔄

**Goal:** Test HIP kernel with `--cuda-lower` flag
**Duration:** 2 hours
**Status:** Ready to start

**Files:**
- Test kernel: To be created
- Results: To be documented

#### Phase 2B: GPUToVortexLLVM Pass ⏳ Planned

**Goal:** Implement custom MLIR pass for GPU → Vortex mapping
**Duration:** 2-3 weeks
**Status:** Not started

**Planned Location:** [phase2-compiler/](phase2-compiler/) (to be created)

**Work:**
- Design Vortex dialect (optional)
- Implement GPUToVortexPass (~500 lines)
- Map GPU operations to Vortex runtime calls:
  - `gpu.thread_id` → `call @vx_thread_id()`
  - `gpu.block_id` → compute from vx_warp_id()
  - `gpu.barrier` → `call @vx_barrier()`

#### Phase 2C: Metadata Integration ⏳ Planned

**Goal:** Integrate metadata extraction with compiler
**Duration:** 1 week
**Status:** Not started

**Work:**
- Extract metadata from MLIR function signatures
- Integrate with Phase 1 metadata generator
- Generate kernel registration code automatically

---

### Phase 3: Runtime Library ⏳ PLANNED

**Status:** Planned, not started
**Duration:** 3-4 weeks (estimated)

**Planned Location:** [phase3-runtime/](phase3-runtime/)

**Scope:**
- Host-side HIP API implementation
- Memory management (hipMalloc, hipMemcpy, hipFree)
- Device management (hipGetDeviceCount, hipSetDevice, etc.)
- Kernel execution (hipLaunchKernel, hipDeviceSynchronize)
- Error handling

---

## 🗂️ Repository Structure

```
vortex_hip/
├── README.md                          # Project overview ⭐
├── INDEX.md                           # This file
├── docs/PHASES_OVERVIEW.md                 # Phase details
├── docs/PHASE1_COMPLETE.md                 # Phase 1 report
├── CONTRIBUTING.md                    # Git guidelines
│
├── docs/phase2-polygeist/                         # Phase 2: C++ → SCF 🔄
│   ├── POLYGEIST_STATUS.md            # ⭐ CURRENT STATUS
│   ├── hip_minimal.h                  # HIP header
│   ├── HIP_SUPPORT_INVESTIGATION.md   # HIP analysis
│   ├── build/bin/cgeist               # Tool (202MB)
│   └── [Polygeist source]
│
├── phase1-metadata/                   # Phase 1 ✅
│   ├── hip_metadata_gen.py            # Generator
│   ├── test_kernels/                  # Test HIP code
│   └── results/                       # Validation
│
├── phase2-compiler/                   # Phase 2B/2C ⏳
│   └── [To be created]
│
├── phase3-runtime/                    # Phase 3 ⏳
│   └── [To be created]
│
├── docs/
│   ├── implementation/                # Guides
│   ├── reference/                     # Architecture
│       └── old-compiler-approaches/   # Rejected approaches
│
├── vortex/                            # Vortex GPU (submodule)
├── llvm-vortex/                       # LLVM backend (submodule)
```

---

## 🎯 By Use Case

### I want to...

#### ...understand what's happening now
→ [docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)

#### ...understand the overall project
→ [README.md](README.md)

#### ...see what's been completed
→ [docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md)

#### ...understand the phases
→ [docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)

#### ...understand HIP → Vortex API mapping
→ [docs/implementation/HIP-TO-VORTEX-API-MAPPING.md](docs/implementation/HIP-TO-VORTEX-API-MAPPING.md)

#### ...understand Vortex GPU capabilities
→ [docs/reference/VORTEX-ARCHITECTURE.md](docs/reference/VORTEX-ARCHITECTURE.md)

#### ...understand the compiler design
→ [docs/implementation/COMPILER_INFRASTRUCTURE.md](docs/implementation/COMPILER_INFRASTRUCTURE.md)

#### ...understand metadata generation
→ [docs/implementation/COMPILER_METADATA_GENERATION.md](docs/implementation/COMPILER_METADATA_GENERATION.md)

#### ...see why other approaches were rejected

#### ...contribute to the project
→ [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📊 Implementation Status

### Overall Progress

| Phase | Status | Progress | Timeline |
|-------|--------|----------|----------|
| Phase 1 | ✅ Complete | 100% | 2 weeks (done) |
| Phase 2A | 🔄 Current | 0% | 2 hours |
| Phase 2B | ⏳ Planned | 0% | 2-3 weeks |
| Phase 2C | ⏳ Planned | 0% | 1 week |
| Phase 3 | ⏳ Planned | 0% | 3-4 weeks |
| **Total** | **~20% done** | **20%** | **~10 weeks** |

### Phase 2 Subtask Status

| Subtask | Status | Duration | Dependencies |
|---------|--------|----------|--------------|
| Polygeist build | ✅ Done | 1 day | None |
| Polygeist validation | ✅ Done | 2 days | Build |
| HIP investigation | ✅ Done | 1 day | Build |
| HIP testing (2A) | 🔄 Current | 2 hours | Validation |
| GPUToVortexLLVM (2B) | ⏳ Next | 2-3 weeks | 2A |
| Metadata integration (2C) | ⏳ Waiting | 1 week | 2B |

---

## 🔑 Key Features & Capabilities

### Completed (Phase 1)
✅ **Metadata Generation**
- DWARF debug info parsing
- Automatic offset calculation
- Pointer detection
- Registration code generation

### In Progress (Phase 2)
🔄 **Compiler Infrastructure**
- Polygeist built and validated
- C++ → SCF conversion working
- HIP support investigated
- Ready for custom passes

### Planned (Phase 2 & 3)
⏳ **GPU → Vortex Lowering**
⏳ **Runtime Library**
⏳ **End-to-end Pipeline**

---

## 📈 Recent Updates

**November 10, 2025:**
- ✅ Documentation consolidated and organized
- ✅ README updated with current status
- ✅ INDEX updated with clear navigation

**November 9-10, 2025:**
- ✅ Polygeist built successfully
- ✅ Validation complete (5/5 tests)
- ✅ HIP support investigated
- ✅ Architecture validated

**Early November 2025:**
- ✅ Phase 1 completed
- ✅ Metadata generator working
- ✅ Approach selection (Polygeist chosen)

---

## 🔗 External Resources

### Official Documentation
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [Polygeist Paper](https://mlir.llvm.org/OpenMeetings/2021-01-14-Polygeist.pdf)
- [Polygeist GitHub](https://github.com/llvm/Polygeist)

### Related Projects
- [Vortex GPU](https://github.com/vortexgpgpu/vortex) - Target platform
- [chipStar](https://github.com/CHIP-SPV/chipStar) - Alternative HIP implementation
- [LLVM Project](https://llvm.org/) - Compiler infrastructure

---

## 🎓 Learning Path

**For New Contributors:**

1. **Start Here:** [README.md](README.md)
   - Understand project goals
   - See current status
   - Understand architecture

2. **Phase Overview:** [docs/PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)
   - Understand implementation strategy
   - See phase dependencies
   - Understand timeline

3. **Current Work:** [docs/phase2-polygeist/STATUS.md](docs/phase2-polygeist/STATUS.md)
   - See what's being worked on
   - Understand next steps
   - Find parallel work opportunities

4. **Technical Deep Dive:**
   - [docs/implementation/](docs/implementation/) - Implementation guides
   - [docs/reference/](docs/reference/) - Architecture specs
   - [docs/PHASE1_COMPLETE.md](docs/PHASE1_COMPLETE.md) - Completed work

5. **Historical Context:**

---

## 📝 Documentation Standards

All documentation follows guidelines in [CONTRIBUTING.md](CONTRIBUTING.md):
- Markdown format
- Clear headings
- Code examples
- Status indicators (✅ 🔄 ⏳ ❌)
- Last updated dates

---

**This index is maintained to help navigate the growing documentation. For questions or suggestions, see [CONTRIBUTING.md](CONTRIBUTING.md).**
