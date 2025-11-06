# HIP Implementation for Vortex GPU

## Project Overview

This repository contains documentation and analysis for implementing HIP (Heterogeneous-computing Interface for Portability) support on the **Vortex RISC-V GPU**.

**IMPORTANT DISCOVERY:** Vortex **already has OpenCL 1.2 support** via POCL, and chipStar **already has an OpenCL backend**. This enables a **hybrid approach**:

1. **Base Layer (Week 1):** Use chipStar + OpenCL for standard HIP (90% coverage, minimal work)
2. **Extension Layer (Week 2-3):** Add Vortex-specific intrinsics (warp shuffles, voting, etc.)
3. **Optional Direct Backend (Week 4+):** Bypass OpenCL for maximum performance if needed

### Goal

Enable standard HIP/CUDA applications on Vortex with optional access to Vortex-specific optimizations through a tiered approach.

## Quick Start: HIP on Vortex

### Prerequisites

**Note:** Throughout this guide, configure the following environment variables for your specific installation:
- `VORTEX_ROOT` - Path to your Vortex installation
- `HIP_INSTALL` - Path to your chipStar/HIP installation

```bash
# 1. Vortex built with OpenCL support
cd ${VORTEX_ROOT}/build
source ./ci/toolchain_env.sh

# 2. chipStar built with OpenCL backend
cd /path/to/chipStar
mkdir build && cd build
cmake .. -DCHIP_BUILD_OPENCL=ON -DCMAKE_INSTALL_PREFIX=${HIP_INSTALL}
make -j$(nproc)
make install
```

### Run HIP Program
```bash
# Set environment variables (configure paths for your installation)
export VORTEX_ROOT=/path/to/vortex
export HIP_INSTALL=/path/to/hip/install

# Point chipStar to Vortex's OpenCL
export OCL_ICD_VENDORS=${VORTEX_ROOT}/runtime/pocl/vendors

# Compile HIP program
# IMPORTANT: HIP programs must be linked to TWO runtime libraries:
#   1. Vortex runtime library (${VORTEX_ROOT}/stub/libvortex.so)
#   2. HIP runtime library (${HIP_INSTALL}/lib/libCHIP.so)
${HIP_INSTALL}/bin/hipcc vector_add.hip \
    -L${VORTEX_ROOT}/stub -lvortex \
    -o vector_add

# Run
./vector_add
```

**That's it!** HIP programs now run on Vortex through chipStar → OpenCL → POCL → Vortex.

---

## Compilation and Linking Details

### Required Libraries

HIP programs on Vortex require linking to **TWO runtime libraries**:

1. **Vortex Runtime Library** (`libvortex.so`)
   - Location: `${VORTEX_ROOT}/stub/`
   - Purpose: GPU driver and hardware access
   - Provides: `vx_dev_open()`, `vx_mem_alloc()`, `vx_start()`, etc.

2. **HIP Runtime Library** (`libCHIP.so`)
   - Location: `${HIP_INSTALL}/lib/`
   - Purpose: HIP API implementation
   - Provides: `hipMalloc()`, `hipMemcpy()`, `hipLaunchKernel()`, etc.

### Complete Example

```cpp
// vector_add.hip
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // Copy to device
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock),
                       0, 0, d_a, d_b, d_c, N);

    // Copy result back
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }

    printf("%s\n", success ? "PASSED" : "FAILED");

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return success ? 0 : 1;
}
```

### Compilation Steps

```bash
# Step 1: Set environment (configure paths for your installation)
export VORTEX_ROOT=/path/to/vortex
export HIP_INSTALL=/path/to/hip/install
export OCL_ICD_VENDORS=${VORTEX_ROOT}/runtime/pocl/vendors
export PATH=${HIP_INSTALL}/bin:$PATH

# Step 2: Compile with hipcc (automatic linking)
# The hipcc wrapper handles most of the compilation, but we need to
# explicitly link to the Vortex runtime library
hipcc vector_add.hip \
    -L${VORTEX_ROOT}/stub -lvortex \
    -o vector_add

# Step 3: Run
./vector_add
```

### Manual Compilation (Alternative)

If you need more control, you can compile manually:

```bash
# Compile host code
clang++ -c vector_add.hip \
    -I${HIP_INSTALL}/include \
    -D__HIP_PLATFORM_SPIRV__ \
    -o vector_add.o

# Compile device code (handled by hipcc internally)
# This step involves: HIP → LLVM IR → SPIR-V → embedded in object file

# Link with both runtime libraries
clang++ vector_add.o \
    -L${VORTEX_ROOT}/stub -lvortex \
    -L${HIP_INSTALL}/lib -lCHIP \
    -Wl,-rpath,${VORTEX_ROOT}/stub:${HIP_INSTALL}/lib \
    -o vector_add
```

### Troubleshooting

**Error: `libvortex.so: cannot open shared object file`**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${VORTEX_ROOT}/stub:${HIP_INSTALL}/lib:$LD_LIBRARY_PATH
```

**Error: `libCHIP.so: cannot open shared object file`**
```bash
# Ensure HIP installation is correct
ls ${HIP_INSTALL}/lib/libCHIP.so
# If missing, rebuild chipStar
```

**Error: `No OpenCL platforms found`**
```bash
# Ensure Vortex OpenCL is configured
export OCL_ICD_VENDORS=${VORTEX_ROOT}/runtime/pocl/vendors
# Verify
clinfo
```

---

## Documentation Overview

### Start Here (Implementation)
1. **[docs/reference/VORTEX-ARCHITECTURE.md](docs/reference/VORTEX-ARCHITECTURE.md)** - Vortex GPU capabilities and runtime API
2. **[docs/implementation/HYBRID-APPROACH.md](docs/implementation/HYBRID-APPROACH.md)** - **Recommended implementation strategy**
   - Tier 1: OpenCL base (1 week)
   - Tier 2: Vortex extensions (2 weeks)
   - Tier 3: Direct backend (optional)

### Reference Analysis (Background)
3. **[docs/analysis/CHIPSTAR-RUNTIME-ANALYSIS.md](docs/analysis/CHIPSTAR-RUNTIME-ANALYSIS.md)** - How chipStar works
4. **[docs/analysis/HIP-CPU-ARCHITECTURE-ANALYSIS.md](docs/analysis/HIP-CPU-ARCHITECTURE-ANALYSIS.md)** - Minimal HIP reference
5. **[docs/implementation/IMPLEMENTATION-COMPARISON.md](docs/implementation/IMPLEMENTATION-COMPARISON.md)** - Side-by-side comparisons

### Additional Resources
6. **[docs/analysis/CHIPSTAR-ARCHITECTURE-ANALYSIS.md](docs/analysis/CHIPSTAR-ARCHITECTURE-ANALYSIS.md)** - Complete chipStar details
7. **[docs/implementation/VORTEX-HIP-IMPLEMENTATION-GUIDE.md](docs/implementation/VORTEX-HIP-IMPLEMENTATION-GUIDE.md)** - ~~Custom implementation~~ (Superseded by hybrid approach)

## Referenced Projects

This documentation references the following external projects (not included in this repository):

1. **Vortex** - RISC-V GPU with **existing OpenCL 1.2 support**
   - Runtime API: `vx_dev_open`, `vx_mem_alloc`, `vx_start`, etc.
   - Kernel API: `threadIdx`, `blockIdx`, `__syncthreads()`, `__local_mem()`
   - Intrinsics: warp voting, shuffles, thread control
2. **chipStar** - Complete HIP runtime with OpenCL backend (use as-is!)
3. **HIP-CPU** - CPU-only HIP (reference for API understanding)
4. **hip** - Official HIP headers

These projects should be cloned separately and are analyzed in the documentation files.

---

## Repository Structure

```
vortex_hip/
├── .gitignore                         # Excludes external project directories
├── README.md                          # This file
├── CONTRIBUTING.md                    # Commit and documentation guidelines
│
└── docs/
    ├── SUMMARY.md                     # Project summary
    │
    ├── analysis/                      # Analysis of existing implementations
    │   ├── CHIPSTAR-ARCHITECTURE-ANALYSIS.md
    │   ├── CHIPSTAR-RUNTIME-ANALYSIS.md
    │   ├── CHIPSTAR-LOWERING-ANALYSIS.md
    │   ├── HIP-CPU-ARCHITECTURE-ANALYSIS.md
    │   └── chipstar_analysis.md
    │
    ├── implementation/                # Implementation guides and strategies
    │   ├── HYBRID-APPROACH.md         # Recommended implementation strategy
    │   ├── VORTEX-HIP-IMPLEMENTATION-GUIDE.md
    │   └── IMPLEMENTATION-COMPARISON.md
    │
    └── reference/                     # Reference documentation
        └── VORTEX-ARCHITECTURE.md     # Vortex GPU capabilities and runtime
```

**Note:** The external projects (chipStar, HIP-CPU, hip) are referenced in the documentation but not included in this repository. Clone them separately as needed.

---

## Reference Implementations

### 1. chipStar - SPIR-V Based Runtime

**Type:** Full HIP runtime with SPIR-V intermediate representation

**Key Features:**
- Uses Clang's native HIP support (no compiler modifications)
- Compiles HIP/CUDA → LLVM IR → SPIR-V
- Supports multiple backends (OpenCL, Intel Level Zero)
- Custom LLVM passes for HIP→SPIR-V translation
- JIT compilation via backend drivers

**Architecture:**
```
HIP Application
      ↓
Clang Frontend (HIP support built-in)
      ↓
LLVM IR (device code)
      ↓
chipStar LLVM Passes (HIP transformations)
      ↓
SPIRV-LLVM-Translator
      ↓
SPIR-V Binary (embedded in fat binary)
      ↓
chipStar Runtime (libCHIP.so)
      ↓
Backend: OpenCL or Level Zero
      ↓
Hardware (GPU)
```

**Complexity:** ~50,000+ lines of code

**Relevant Analysis:**
- [docs/analysis/CHIPSTAR-ARCHITECTURE-ANALYSIS.md](docs/analysis/CHIPSTAR-ARCHITECTURE-ANALYSIS.md) - Complete architecture
- [docs/analysis/CHIPSTAR-RUNTIME-ANALYSIS.md](docs/analysis/CHIPSTAR-RUNTIME-ANALYSIS.md) - Runtime details
- [docs/analysis/chipstar_analysis.md](docs/analysis/chipstar_analysis.md) - Comprehensive analysis

### 2. HIP-CPU - Header-Only Implementation

**Type:** Pure CPU implementation (no GPU required)

**Key Features:**
- Header-only C++ library (~2,000 lines)
- Requires only C++17 standard library
- Uses `std::execution::par_unseq` for parallelism
- Cooperative multitasking for barriers (fibers)
- Unified memory model (all pointers are CPU pointers)

**Architecture:**
```
HIP Application
      ↓
#include <hip/hip_runtime.h>
      ↓
Inline C++ Implementation:
  - hipMalloc → malloc()
  - hipMemcpy → memcpy()
  - kernel<<<>>> → std::for_each(par_unseq)
  - __syncthreads() → fiber context switch
      ↓
C++17 Parallel Algorithms
      ↓
CPU threads (8-16 worker threads)
```

**Complexity:** ~2,000 lines of code

**Thread Mapping:**
- HIP Threads: 1,000,000
- OS Threads: 9-17
- Ratio: ~65,000:1

**Relevant Analysis:**
- [docs/analysis/HIP-CPU-ARCHITECTURE-ANALYSIS.md](docs/analysis/HIP-CPU-ARCHITECTURE-ANALYSIS.md) - Complete implementation details

### 3. Official HIP Headers

**Type:** API definitions only

**Purpose:** Standard HIP API declarations used by all implementations

---

## Implementation Comparison

| Aspect | chipStar | HIP-CPU | Target: Vortex |
|--------|----------|---------|----------------|
| **Target** | SPIR-V GPUs | CPU cores | Vortex RISC-V GPU |
| **Compilation** | Clang → SPIR-V | Direct C++ | Clang → ? |
| **IR Format** | SPIR-V | None (header-only) | TBD (SPIR-V or custom?) |
| **Backend** | OpenCL/Level0 | C++17 stdlib | Vortex driver/runtime |
| **Memory Model** | Host ≠ Device | Unified (CPU RAM) | Host ≠ Device |
| **LOC** | ~50,000+ | ~2,000 | TBD |
| **Barriers** | Hardware | O(n) fibers | Hardware (expected) |
| **JIT Compilation** | Yes (driver) | No | TBD |
| **Performance** | Native GPU | 10-100x slower | Native GPU (goal) |

---

## Key Technical Decisions for Vortex

### 1. Intermediate Representation

**Option A: SPIR-V (chipStar approach)**

Pros:
- Leverage existing Clang HIP support
- Reuse chipStar's LLVM passes
- Standard IR with rich tooling
- Backend-agnostic

Cons:
- Requires SPIR-V support in Vortex
- Complex runtime (SPIR-V parsing/JIT)
- May not map perfectly to RISC-V ISA

**Option B: Custom IR / Direct RISC-V**

Pros:
- Optimal code generation for RISC-V
- No translation overhead
- Full control over compilation

Cons:
- Requires custom compiler modifications
- More implementation work
- Less tooling support

### 2. Memory Management

chipStar uses:
- Separate host and device allocations
- Explicit transfers via DMA-like mechanisms
- OpenCL/Level0 memory APIs

Vortex should consider:
- **Unified virtual addressing** (simplifies pointer handling)
- **Explicit copy APIs** (hipMemcpy) for data movement
- **Cache coherency model** (depending on hardware)

### 3. Kernel Launch Mechanism

chipStar:
```cpp
hipLaunchKernel(func, grid, block, args, shared, stream)
  → Find compiled kernel
  → Setup arguments (marshal to backend)
  → Backend launch (OpenCL: clEnqueueNDRangeKernel)
  → Hardware execution
```

Vortex needs:
- Kernel registration mechanism
- Argument marshalling
- Grid/block to hardware mapping
- Stream/queue management

### 4. Runtime Architecture

Required components:
1. **Device Management** - Detect and initialize Vortex GPU
2. **Memory Manager** - Allocate, free, copy device memory
3. **Module Loader** - Load compiled kernels
4. **Kernel Registry** - Map host pointers to device functions
5. **Launch Manager** - Submit kernels with arguments
6. **Stream/Event System** - Asynchronous execution
7. **Error Handling** - hipError_t codes

---

## Implementation Strategy for Vortex

### Phase 1: Minimal Runtime (Core APIs)

Implement essential HIP APIs:
```cpp
// Device management
hipGetDeviceCount()
hipSetDevice()
hipGetDeviceProperties()

// Memory management
hipMalloc()
hipFree()
hipMemcpy()
hipMemcpyAsync()

// Kernel launch
hipLaunchKernel()
hipDeviceSynchronize()
```

### Phase 2: Compilation Pipeline

Choose one of:

**A. Adapt chipStar for Vortex:**
```
Clang HIP frontend
      ↓
LLVM IR
      ↓
chipStar LLVM passes (reuse)
      ↓
SPIRV-LLVM-Translator
      ↓
SPIR-V → Vortex ISA translator (NEW)
      ↓
Vortex binary
```

**B. Custom Vortex backend:**
```
Clang HIP frontend
      ↓
LLVM IR
      ↓
Custom LLVM passes for Vortex
      ↓
LLVM RISC-V backend (with Vortex extensions)
      ↓
Vortex binary
```

### Phase 3: Advanced Features

- Streams and events
- Shared memory
- Atomic operations
- Texture support (if applicable)
- Multi-GPU support

### Phase 4: Optimization

- Kernel caching
- Fast argument setup
- DMA optimization
- Compute unit scheduling

---

## Lessons from Reference Implementations

### From chipStar

1. **Lazy JIT Compilation** - Defer compilation until first kernel launch
2. **Metadata-Driven Execution** - Extract argument types from IR
3. **Backend Abstraction** - Common interface for different execution engines
4. **Fat Binary Format** - Embed device code in host executable
5. **Registration Pattern** - Compiler-generated stubs register kernels

### From HIP-CPU

1. **Unified Memory Simplicity** - Host = Device pointers (if possible)
2. **Minimal Thread Creation** - Reuse thread pools across launches
3. **Two-Mode Execution** - Fast path (no barriers) vs slow path (with barriers)
4. **Barrier Cost** - Hardware barriers are critical for performance
5. **Shared Memory** - Must be truly shared, not thread-local

### Common Patterns

Both implementations:
- Use SPVRegister-style kernel registry
- Implement hipError_t error codes
- Support asynchronous operations (streams)
- Marshal arguments based on metadata
- Handle kernel launch triple-chevron syntax

---

## Key Files to Study

### chipStar Runtime

| File | Purpose | Lines |
|------|---------|-------|
| `chipStar/src/CHIPBackend.hh` | Backend abstraction | ~1,200 |
| `chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc` | OpenCL implementation | ~3,000 |
| `chipStar/src/SPVRegister.cc` | Module/kernel registry | ~300 |
| `chipStar/src/spirv.cc` | SPIR-V parser | ~4,800 |
| `chipStar/src/CHIPBindings.cc` | Public HIP API | ~6,000 |

### chipStar LLVM Passes

| Pass | Purpose |
|------|---------|
| `HipTextureLowering.cpp` | Convert texture ops → image ops |
| `HipKernelArgSpiller.cpp` | Handle large arguments |
| `HipDynMem.cpp` | Dynamic shared memory |
| `HipGlobalVariables.cpp` | Device global variables |
| `HipAbort.cpp` | Device-side assertions |

### HIP-CPU Runtime

| File | Purpose |
|------|---------|
| `HIP-CPU/include/hip/hip_runtime.h` | Public API |
| `HIP-CPU/src/include/hip/detail/grid_launch.hpp` | Kernel launch |
| `HIP-CPU/src/include/hip/detail/tile.hpp` | Block execution |
| `HIP-CPU/src/include/hip/detail/fiber.hpp` | Barrier implementation |
| `HIP-CPU/src/include/hip/detail/api.hpp` | Memory/device APIs |

---

## Next Steps for Vortex Implementation

### 1. Define Requirements

- [ ] Vortex ISA specification
- [ ] Memory architecture (unified? discrete?)
- [ ] Hardware features (barriers, atomics, shared memory)
- [ ] Driver interface (system calls, MMIO, DMA)
- [ ] Performance targets

### 2. Choose Architecture

- [ ] SPIR-V based (adapt chipStar) or custom?
- [ ] Backend design (OpenCL-like? Custom?)
- [ ] Memory model (unified addressing?)
- [ ] Compilation flow (JIT? AOT? Hybrid?)

### 3. Implement Core Runtime

- [ ] Device initialization
- [ ] Memory allocation/free
- [ ] Kernel loading
- [ ] Launch mechanism
- [ ] Synchronization primitives

### 4. Develop Compiler Support

- [ ] LLVM backend for Vortex (if not exists)
- [ ] HIP-specific transformations
- [ ] Bitcode device library
- [ ] Kernel metadata generation

### 5. Testing and Validation

- [ ] Unit tests (API correctness)
- [ ] Integration tests (simple kernels)
- [ ] Performance benchmarks
- [ ] Compatibility suite (HIP tests)

---

## Building Reference Implementations

### chipStar

```bash
cd chipStar
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHIP_BUILD_OPENCL=ON
make -j$(nproc)
```

### HIP-CPU

Header-only - no build required. Just include:
```cpp
#include <hip/hip_runtime.h>
```

Requires C++17 compiler with parallel algorithm support.

---

## Resources

### Documentation
- [chipStar GitHub](https://github.com/CHIP-SPV/chipStar)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [SPIR-V Specification](https://www.khronos.org/registry/spir-v/)

### Analysis Documents in This Repo
- **docs/analysis/CHIPSTAR-ARCHITECTURE-ANALYSIS.md** - Complete chipStar architecture
- **docs/analysis/CHIPSTAR-RUNTIME-ANALYSIS.md** - Runtime loading and execution
- **docs/analysis/CHIPSTAR-LOWERING-ANALYSIS.md** - LLVM IR transformations
- **docs/analysis/HIP-CPU-ARCHITECTURE-ANALYSIS.md** - CPU implementation details

### Related Projects
- [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY) - CUDA to HIP conversion
- [HIP Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples)
- [Vortex GPU](https://github.com/vortexgpgpu/vortex) - Target platform

---

## Comparison: chipStar vs Vortex Requirements

### Memory Bandwidth
- **chipStar (OpenCL GPU):** 100-1000 GB/s
- **Vortex (RISC-V):** TBD - likely lower
- **Impact:** May need different optimization strategies

### Thread Count
- **chipStar (Modern GPU):** 1,000-10,000 threads
- **Vortex:** TBD
- **Impact:** Affects block size recommendations

### Barrier Performance
- **chipStar (GPU):** Hardware barriers (essentially free)
- **HIP-CPU:** O(blockDim) context switches (expensive)
- **Vortex:** Hardware barriers expected
- **Impact:** Critical for shared memory patterns

### Memory Hierarchy
- **chipStar (GPU):** L1, L2, LDS (shared), global
- **Vortex:** TBD - RISC-V cache hierarchy
- **Impact:** Shared memory usage patterns

---

## License

This repository contains multiple projects with different licenses:
- **chipStar:** MIT License
- **HIP-CPU:** MIT License
- **hip:** MIT License

See individual LICENSE files in each subdirectory.

---

## Contact and Contributions

For questions about Vortex HIP implementation, please contact the Vortex team.

To study the reference implementations:
1. Read the analysis documents
2. Explore the source code
3. Build and run examples
4. Compare architectural approaches

---

**Last Updated:** 2025-10-29
**Status:** Analysis and planning phase
**Next Milestone:** Define Vortex HIP architecture
