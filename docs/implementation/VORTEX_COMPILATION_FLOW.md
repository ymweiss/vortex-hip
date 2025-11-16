# Vortex Compilation Flow and Architecture Analysis

## Document Purpose

This document describes how Vortex compiles GPU kernels and integrates with the compiler toolchain. Understanding this flow is critical for implementing the HIP→Vortex compilation pipeline.

---

## 1. COMPILER TOOLCHAIN INTEGRATION

### 1.1 Compiler Components and Locations

**LLVM-Vortex Backend:**
- **Location**: `llvm-vortex/`
- **Purpose**: LLVM compiler with Vortex-specific RISC-V target extensions
- **Base**: LLVM 14+ monorepo (llvm, clang, compiler-rt, etc.)
- **Target**: RISC-V with custom `+vortex` feature flag

**RISC-V GNU Toolchain:**
- **Location**: `$(TOOLDIR)/riscv32-gnu-toolchain` or `riscv64-gnu-toolchain`
- **Purpose**: GCC cross-compiler, binutils, and C library for final linking
- **Prefix**: `riscv32-unknown-elf` or `riscv64-unknown-elf`

**POCL (Portable OpenCL):**
- **Location**: `$(TOOLDIR)/pocl`
- **Purpose**: OpenCL runtime with Vortex device backend
- **Integration**: Uses LLVM-Vortex for code generation

### 1.2 Environment Configuration

From `vortex/build/config.mk`:

```makefile
# Toolchain Locations
TOOLDIR ?= $(HOME)/tools
LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv$(XLEN)-gnu-toolchain
RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

# Vortex Specific
VORTEX_HOME ?= vortex
XLEN ?= 32  # 32-bit or 64-bit

# C Library
LIBC_VORTEX ?= $(TOOLDIR)/libc$(XLEN)
LIBCRT_VORTEX ?= $(TOOLDIR)/libcrt$(XLEN)
```

### 1.3 Toolchain Installation

**Installation Script**: `vortex/ci/toolchain_install.sh.in`

The script downloads and installs prebuilt toolchains:

```bash
# Install LLVM-Vortex
./toolchain_install.sh --llvm

# Install RISC-V GNU toolchain
./toolchain_install.sh --riscv32  # or --riscv64

# Install POCL
./toolchain_install.sh --pocl

# Install runtime libraries
./toolchain_install.sh --libcrt32 --libc32

# Install everything
./toolchain_install.sh --all
```

**Prebuilt Repository**: `https://github.com/vortexgpgpu/vortex-toolchain-prebuilt`

---

## 2. COMPILATION PIPELINE DETAILS

### 2.1 Complete Flow

```
OpenCL/GPU Source (.cl)
         |
         v
   POCL Frontend
         |  (Uses Clang from LLVM-Vortex)
         v
   LLVM IR Generation
         |  (with OpenCL builtins)
         v
   LLVM Optimization Passes
         |  (Standard LLVM -O3)
         v
   Vortex-Specific Passes
         |  - Branch divergence handling
         |  - Vortex intrinsic lowering
         |  - Custom instruction selection
         v
   LLVM RISC-V Backend
         |  Target: riscv32/64 +vortex +zicond
         v
   RISC-V Assembly (.s)
         |
         v
   RISC-V GCC Assembler
         |
         v
   Object Files (.o)
         |
         v
   RISC-V GCC Linker
         |  Links with:
         |  - libvortex.a (GPU runtime)
         |  - libc.a, libm.a (standard libs)
         |  - libclang_rt.builtins-riscv32.a
         |  - Custom linker script (link32.ld)
         v
   ELF Binary
         |
         v
   vxbin.py Post-Processing
         |  - Extracts LOAD segments
         |  - Creates .vxbin format
         v
   Vortex Binary (.vxbin)
         |
         v
   Upload to Vortex Device
```

### 2.2 Compilation Flags

From `vortex/kernel/Makefile` and test Makefiles:

**Architecture Flags:**
```makefile
# 32-bit
CFLAGS += -march=rv32imaf -mabi=ilp32f

# 64-bit
CFLAGS += -march=rv64imafd -mabi=lp64d
```

**Vortex-Specific Flags:**
```makefile
# Enable Vortex target feature
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex

# Branch divergence handling (optional)
LLVM_CFLAGS += -mllvm -vortex-branch-divergence=0

# Disable loop idiom recognition (prevents memset/memcpy generation)
CFLAGS += -mllvm -disable-loop-idiom-all

# Code model and sysroot
CFLAGS += -mcmodel=medany
CFLAGS += --sysroot=$(RISCV_SYSROOT)
CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
```

**Optimization and Code Generation:**
```makefile
CFLAGS += -O3
CFLAGS += -fno-rtti -fno-exceptions
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += -nostartfiles -nostdlib  # For kernels
```

**Linker Flags:**
```makefile
LDFLAGS += -Wl,-Bstatic,--gc-sections  # Static linking, remove unused
LDFLAGS += -T$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld
LDFLAGS += --defsym=STARTUP_ADDR=$(STARTUP_ADDR)
LDFLAGS += $(VORTEX_HOME)/kernel/libvortex.a
LDFLAGS += -lm -lc
```

### 2.3 Kernel Library Build

**Location**: `vortex/kernel/`

**Makefile**: `vortex/kernel/Makefile`

**Build Process:**
```makefile
# Source files for libvortex.a
SRCS = vx_start.S       # Startup code (wspawn, TLS setup, main)
       vx_syscalls.c    # System call implementations
       vx_print.S       # Low-level print functions
       vx_print.c       # Printf implementation
       tinyprintf.c     # Lightweight printf
       vx_spawn.c       # Grid/block spawn framework
       vx_serial.S      # Serial I/O
       vx_perf.c        # Performance counters

# Compiler
CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar

# Build
$(PROJECT).a: $(OBJS)
    $(AR) rcs $@ $^
```

**Note**: Can also use LLVM toolchain (commented out in Makefile):
```makefile
#CC = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
#AR = $(LLVM_VORTEX)/bin/llvm-ar
```

---

## 3. VORTEX ISA AND PRIMITIVES

### 3.1 Base ISA

**RISC-V Standard Extensions:**
- **RV32IMAF** (32-bit) or **RV64IMAFD** (64-bit)
  - I: Integer base instructions
  - M: Integer multiply/divide
  - A: Atomic operations (optional, for synchronization)
  - F: Single-precision floating-point
  - D: Double-precision floating-point (64-bit only)

**Vortex Custom Extensions:**
- **+vortex**: Custom GPU instructions in RISC-V CUSTOM-0 opcode space (0x0B)
- **+zicond**: Conditional operations extension (RISC-V ratified)

### 3.2 Vortex GPU Instructions

**Header**: `vortex/kernel/include/vx_intrinsics.h`

**Thread Control:**
```c
void vx_tmc(int thread_mask);           // Thread mask control
void vx_tmc_zero();                     // Disable all threads
void vx_tmc_one();                      // Enable only thread 0

void vx_wspawn(int num_warps, void (*func_ptr)());  // Spawn warps

void vx_pred(int condition, int mask);  // Thread predication
int vx_split(int predicate);            // Divergence: save state
void vx_join(int stack_ptr);            // Convergence: restore state
```

**Synchronization:**
```c
void vx_barrier(int barrier_id, int num_warps);  // Warp barrier
void vx_fence();                                 // Memory fence
```

**Thread ID Queries (CSR reads):**
```c
int vx_thread_id();      // Thread within warp (CSR 0xCC0)
int vx_warp_id();        // Warp within core (CSR 0xCC1)
int vx_core_id();        // Core ID (CSR 0xCC2)
int vx_num_threads();    // Threads per warp (CSR 0xFC0)
int vx_num_warps();      // Warps per core (CSR 0xFC1)
int vx_num_cores();      // Total cores (CSR 0xFC2)
```

**Warp Collective Operations:**
```c
int vx_vote_all(int predicate);     // All threads true
int vx_vote_any(int predicate);     // Any thread true
int vx_vote_ballot(int predicate);  // Bitmask
int vx_shfl_up/down/bfly/idx(...);  // Warp shuffles
```

### 3.3 Assembly Encoding

All Vortex instructions use `.insn r` format with `RISCV_CUSTOM0 (0x0B)`:

```assembly
# Thread Mask Control
.insn r RISCV_CUSTOM0, 0, 0, x0, rs1, x0

# Warp Spawn
.insn r RISCV_CUSTOM0, 1, 0, x0, rs1, rs2

# Split/Join (divergence)
.insn r RISCV_CUSTOM0, 2, 0, rd, rs1, x0  # split
.insn r RISCV_CUSTOM0, 3, 0, x0, rs1, x0  # join

# Barrier
.insn r RISCV_CUSTOM0, 4, 0, x0, rs1, rs2

# Predication
.insn r RISCV_CUSTOM0, 5, 0, func, rs1, rs2  # func=0/1 for pred/pred_n
```

---

## 4. THREAD/WARP/CORE MODEL

### 4.1 Hierarchy

```
Vortex Device
  └── Clusters (NUM_CLUSTERS)
       └── Cores (NUM_CORES per cluster)
            └── Warps (NUM_WARPS per core, default: 4)
                 └── Threads (NUM_THREADS per warp, default: 4)
```

**Default Configuration**: 1 cluster, 1 core, 4 warps, 4 threads = 16 hardware threads

### 4.2 Mapping to CUDA/HIP

**Spawn API** (`vortex/kernel/include/vx_spawn.h`):

```c
typedef union { uint32_t x, y, z; } dim3_t;

extern __thread dim3_t blockIdx;    // Per-thread
extern __thread dim3_t threadIdx;   // Per-thread
extern dim3_t gridDim;              // Global
extern dim3_t blockDim;             // Global

int vx_spawn_threads(
    uint32_t dimension,
    const uint32_t* grid_dim,     // Number of blocks
    const uint32_t* block_dim,    // Threads per block
    vx_kernel_func_cb kernel_func,
    const void* arg
);
```

**Mapping Table:**

| HIP/CUDA | Vortex | Access Method |
|----------|--------|---------------|
| `gridDim.x` | `gridDim.x` | Global variable |
| `blockDim.x` | `blockDim.x` | Global variable |
| `blockIdx.x` | `blockIdx.x` | `__thread` TLS variable |
| `threadIdx.x` | `threadIdx.x` | `__thread` TLS variable |
| Warp ID | - | `vx_warp_id()` CSR read |
| Thread ID within warp | - | `vx_thread_id()` CSR read |
| `__syncthreads()` | `vx_barrier(__local_group_id, __warps_per_group)` | Intrinsic |

### 4.3 Spawn Implementation

**Source**: `vortex/kernel/src/vx_spawn.c`

The spawn framework distributes grid/blocks across Vortex's warp/thread hierarchy:

```c
// For each block in the grid:
//   Calculate blockIdx.{x,y,z}
//
//   If block has multiple threads:
//     Calculate warps_per_block = ceil(threads_per_block / threads_per_warp)
//     Spawn warps_per_block warps
//     Each warp calculates its threadIdx.{x,y,z}
//
//   If block has one thread:
//     One thread processes entire block
//     threadIdx = {0,0,0}
```

**Key Insight**: The spawn framework **automatically handles** the mapping from HIP's grid/block model to Vortex's warp/thread model. User kernels just read `blockIdx` and `threadIdx`.

---

## 5. MEMORY MODEL

### 5.1 Address Spaces

**Memory Map** (from `vortex/build/hw/VX_config.h`):

```
32-bit:
  IO_BASE_ADDR:        0x00000040
  USER_BASE_ADDR:      0x00010000  (User programs start here)
  STARTUP_ADDR:        0x80000000  (Kernel entry point)
  LMEM_BASE_ADDR:      0xFFFF0000  (Local/shared memory)
  STACK_BASE_ADDR:     0xFFFF0000  (Stack region)

64-bit:
  IO_BASE_ADDR:        0x000000040
  USER_BASE_ADDR:      0x000010000
  STARTUP_ADDR:        0x080000000
  LMEM_BASE_ADDR:      0x1FFFF0000
  STACK_BASE_ADDR:     0x1FFFF0000
```

**HIP/OpenCL Qualifiers:**

| Qualifier | Vortex Implementation |
|-----------|----------------------|
| `__global` / (default) | Standard memory (cached via L1/L2/L3) |
| `__shared__` / `__local` | LMEM (scratchpad per core) |
| `__constant__` | `.rodata` section (cached, read-only) |

**Local Memory (LMEM):**
- Base address: `CSR VX_CSR_LOCAL_MEM_BASE (0xFC3)`
- Size: Configurable, default 16 KB per core
- Allocation: `__local_mem(size)` macro offsets by `__local_group_id`

```c
#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)
```

### 5.2 Cache Hierarchy

```
Core
 ├── Icache (Instruction cache, 16KB default)
 └── Dcache (Data cache, 16KB default)
      └── L2 Cache (1MB default, shared across socket)
           └── L3 Cache (2MB default, shared across cluster)
                └── Main Memory
```

**Cache Properties:**
- Write-through with MSHR (Miss Status Holding Registers)
- Non-blocking pipeline
- Multi-bank parallelism for high throughput
- Coalescing in LSU (Load-Store Unit)

---

## 6. BINARY FORMAT AND LOADING

### 6.1 Vortex Binary Format (.vxbin)

**Tool**: `vortex/kernel/scripts/vxbin.py`

**Format:**
```
+---------------------------+
| min_vma (8 bytes, uint64) |  Minimum virtual address
+---------------------------+
| max_vma (8 bytes, uint64) |  Maximum virtual address
+---------------------------+
| Binary data (N bytes)     |  Raw binary from ELF LOAD segments
+---------------------------+
```

**Creation Process:**
```bash
# 1. Extract LOAD segments from ELF
readelf -l kernel.elf | grep LOAD

# 2. Determine VMA range (min_vma, max_vma)
# 3. Extract binary with objcopy
objcopy -O binary kernel.elf kernel.bin

# 4. Create .vxbin with header
vxbin.py kernel.elf kernel.vxbin
```

### 6.2 Linker Script

**Location**: `vortex/kernel/scripts/link32.ld` (or `link64.ld`)

**Key Sections:**
```ld
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = STARTUP_ADDR, LENGTH = ...
}

SECTIONS {
    .init    : { *(.init) }       # Initialization
    .text    : { *(.text*) }      # Code
    .rodata  : { *(.rodata*) }    # Read-only data
    .data    : { *(.data*) }      # Initialized data
    .bss     : { *(.bss*) }       # Uninitialized data
    .tdata   : { *(.tdata*) }     # Thread-local initialized
    .tbss    : { *(.tbss*) }      # Thread-local uninitialized
}
```

**Startup Address:**
- Defined via `--defsym=STARTUP_ADDR=0x80000000`
- Where `_start` is placed in memory

### 6.3 Runtime Loading

**API** (`vortex/runtime/include/vortex.h`):

```c
// Upload kernel binary to device
int vx_upload_kernel_file(vx_device_h device,
                          const char* filename,
                          vx_buffer_h* buffer);

// Start kernel execution
int vx_start(vx_device_h device,
            vx_buffer_h kernel,
            vx_buffer_h arguments);

// Wait for completion
int vx_ready_wait(vx_device_h device, uint64_t timeout);
```

---

## 7. COMPILER INTEGRATION WORKFLOW

### 7.1 Example: OpenCL Kernel Compilation

**Test Location**: `vortex/tests/opencl/vecadd/`

**Makefile** (`common.mk`):

```makefile
# Include Vortex configuration
include $(VORTEX_HOME)/build/config.mk

# Compile flags
VX_CFLAGS = -march=rv32imaf -mabi=ilp32f
VX_CFLAGS += -Xclang -target-feature -Xclang +vortex
VX_CFLAGS += -O3 -mcmodel=medany
VX_CFLAGS += --sysroot=$(RISCV_SYSROOT)
VX_CFLAGS += -fno-rtti -fno-exceptions
VX_CFLAGS += -nostartfiles -nostdlib

# Link flags
VX_LDFLAGS = -Wl,-Bstatic,--gc-sections
VX_LDFLAGS += -T$(VORTEX_HOME)/kernel/scripts/link32.ld
VX_LDFLAGS += $(VORTEX_HOME)/kernel/libvortex.a

# POCL environment for runtime compilation
POCL_CC_FLAGS = LLVM_PREFIX=$(LLVM_VORTEX)
POCL_CC_FLAGS += POCL_VORTEX_XLEN=32
POCL_CC_FLAGS += POCL_VORTEX_BINTOOL="$(VXBIN_TOOL)"
POCL_CC_FLAGS += POCL_VORTEX_CFLAGS="$(VX_CFLAGS)"
POCL_CC_FLAGS += POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)"
```

**Compilation at Runtime:**
1. Application calls `clBuildProgram()`
2. POCL Vortex backend invokes LLVM-Vortex
3. Compiles with `VX_CFLAGS`, links with `VX_LDFLAGS`
4. Runs `vxbin.py` to create `.vxbin`
5. Returns compiled kernel to application

### 7.2 HIP Integration Points

For HIP → Vortex compilation, integrate at these points:

**Option A: Runtime Compilation (like POCL)**
```
HIP Source → Polygeist → GPU Dialect IR → LLVM IR →
  LLVM-Vortex Backend → RISC-V Assembly → Link → .vxbin
```

**Option B: Ahead-of-Time Compilation**
```
HIP Source → Offline Compiler → .vxbin →
  Embed in application → Runtime loads .vxbin
```

**Recommended**: Hybrid approach
- Use Polygeist for GPU dialect generation (offline or runtime)
- Custom pass to lower GPU dialect → Vortex spawn API
- LLVM-Vortex backend for code generation
- Standard linking with `libvortex.a`

---

## 8. KEY INTEGRATION REQUIREMENTS FOR HIP

### 8.1 Compiler Modifications Needed

1. **HIP Intrinsic Lowering**:
   - Lower `threadIdx/blockIdx/blockDim/gridDim` to TLS reads or spawn variables
   - Convert `__syncthreads()` to `vx_barrier()`
   - Map `__shared__` to LMEM allocation

2. **Kernel Wrapping**:
   - Generate spawn framework wrapper (like `vx_spawn.c`)
   - Set up `blockIdx/threadIdx` calculation per iteration
   - Handle grid/block dimension loops

3. **Shared Memory Allocation**:
   - Calculate total `__shared__` memory per block
   - Generate `__local_mem(size)` allocation calls
   - Offset by `__local_group_id`

4. **Divergence Handling** (optional):
   - Insert `vx_split()/vx_join()` for divergent branches
   - Or rely on LLVM's existing infrastructure

### 8.2 Build System Integration

**Add to Vortex Makefiles**:

```makefile
# HIP Compilation Flags
HIP_CFLAGS = $(VX_CFLAGS)
HIP_CFLAGS += -I$(HIP_INCLUDE_PATH)
HIP_CFLAGS += --hip-device-lib-path=...

# HIP Lowering
HIP_LDFLAGS = $(VX_LDFLAGS)

# Compile HIP kernel
%.hip.o: %.hip
    $(LLVM_VORTEX)/bin/clang $(HIP_CFLAGS) -c $< -o $@

# Link
kernel.elf: kernel.hip.o
    $(LLVM_VORTEX)/bin/clang $(HIP_LDFLAGS) $^ -o $@

# Package
kernel.vxbin: kernel.elf
    $(VXBIN_TOOL) $< $@
```

### 8.3 Runtime Integration

**Update Vortex Runtime**:

```c
// New HIP-specific runtime API
int vx_hip_launch_kernel(
    vx_device_h device,
    vx_buffer_h kernel,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem
);
```

**Internally calls**:
```c
vx_spawn_threads(3, gridDim, blockDim, kernel_func, args);
```

---

## 9. SUMMARY AND RECOMMENDATIONS

### 9.1 Key Takeaways

1. **Vortex uses standard LLVM/RISC-V toolchain** with custom `+vortex` feature
2. **libvortex.a provides spawn framework** that handles grid/block mapping
3. **Binary format (.vxbin)** is simple: VMA header + raw binary
4. **POCL demonstrates complete integration** for OpenCL, can be model for HIP
5. **LLVM-Vortex backend is at** `llvm-vortex/`

### 9.2 HIP Compilation Strategy

**Phase 1: Leverage Polygeist for Parsing**
- Use Polygeist to parse HIP syntax → GPU dialect IR
- Manual observation of patterns (already done)

**Phase 2: Custom Lowering Pass**
- Create MLIR pass to lower GPU dialect → Vortex spawn API calls
- Map GPU operations to Vortex intrinsics
- Generate LLVM IR with Vortex-specific constructs

**Phase 3: Use LLVM-Vortex Backend**
- Feed LLVM IR to existing LLVM-Vortex backend
- Backend handles RISC-V code generation with `+vortex` features
- Link with `libvortex.a` using standard Vortex build process

**Phase 4: Runtime Integration**
- Create HIP runtime wrapper (`hip_runtime.h` → Vortex API)
- Use existing Vortex device management
- Upload .vxbin kernels using `vx_upload_kernel_file()`

This approach **reuses maximum infrastructure** while adding HIP-specific lowering at the right abstraction layer (MLIR/GPU dialect).

---

## File Reference Summary

**Compiler Toolchain:**
- `llvm-vortex/` - LLVM backend
- `vortex/build/config.mk` - Build configuration
- `vortex/ci/toolchain_install.sh.in` - Toolchain setup

**Kernel Library:**
- `vortex/kernel/Makefile` - libvortex.a build
- `vortex/kernel/src/vx_spawn.c` - Spawn framework
- `vortex/kernel/include/vx_intrinsics.h` - GPU primitives

**Linker and Binary:**
- `vortex/kernel/scripts/link32.ld` - Linker script
- `vortex/kernel/scripts/vxbin.py` - Binary packager

**Examples:**
- `vortex/tests/opencl/` - OpenCL test kernels
- `vortex/tests/opencl/common.mk` - Build example
