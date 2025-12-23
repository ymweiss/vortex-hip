# LLVM IR to Vortex Binary: Integration Plan

This document describes how to integrate Polygeist's LLVM IR output into the standard Vortex compilation flow to produce executable kernel binaries.

## Overview

After Polygeist lowers HIP/CUDA kernels through the GPU dialect and GPUToVortex pass, the output is LLVM IR with Vortex intrinsic calls. This IR must then be compiled to a Vortex binary using the llvm-vortex toolchain.

```
Polygeist Output          llvm-vortex             Final Output
─────────────────    ───────────────────    ─────────────────
kernel.ll/kernel.bc  →  clang/llc/lld   →   kernel.vxbin
(LLVM IR with           (RV32 + Vortex      (Loadable by
 vx_* calls)             extensions)         Vortex runtime)
```

## Compilation Pipeline

### Step 1: Polygeist GPU Dialect to LLVM IR

```bash
# Generate GPU dialect MLIR from HIP source
cgeist --cuda-path=/usr/local/cuda \
       --cuda-gpu-arch=sm_70 \
       -I/usr/local/cuda/include \
       kernel.hip \
       -O2 --function='*' -S --memref-fullrank \
       -o kernel.mlir

# Lower GPU dialect to Vortex intrinsics, then to LLVM dialect
polygeist-opt --convert-gpu-to-vortex \
              --gpu-to-llvm \
              --convert-func-to-llvm \
              --reconcile-unrealized-casts \
              kernel.mlir -o kernel.llvm.mlir

# Translate MLIR LLVM dialect to LLVM IR
mlir-translate --mlir-to-llvmir kernel.llvm.mlir -o kernel.ll
```

### Step 2: LLVM IR to RISC-V Object File

Use llvm-vortex's clang to compile the LLVM IR:

```bash
LLVM_VORTEX=/path/to/llvm-vortex
VORTEX_HOME=/path/to/vortex

$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -mcmodel=medany \
    -O3 \
    -fno-rtti -fno-exceptions \
    -fdata-sections -ffunction-sections \
    -Xclang -target-feature -Xclang +vortex \
    -Xclang -target-feature -Xclang +zicond \
    -mllvm -disable-loop-idiom-all \
    -I$VORTEX_HOME/kernel/include \
    -c kernel.ll \
    -o kernel.o
```

**Key flags:**
| Flag | Purpose |
|------|---------|
| `-march=rv32imaf` | RV32 with integer, multiply, atomic, float |
| `-mabi=ilp32f` | 32-bit ABI with hardware floats |
| `+vortex` | Enable Vortex custom instructions (barriers, thread control) |
| `+zicond` | Enable conditional zero instructions |
| `-mcmodel=medany` | Position-independent addressing |
| `-fno-rtti -fno-exceptions` | Disable C++ runtime features (bare metal) |

### Step 3: Link with Vortex Runtime

```bash
LIBC_VORTEX=/path/to/libc32
LIBCRT_VORTEX=/path/to/libcrt32

$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -nostartfiles -nostdlib \
    -Wl,-Bstatic,--gc-sections \
    -Wl,-T,$VORTEX_HOME/kernel/scripts/link32.ld \
    -Wl,--defsym=STARTUP_ADDR=0x80000000 \
    kernel.o \
    $VORTEX_HOME/build/kernel/libvortex.a \
    -L$LIBC_VORTEX/lib -lm -lc \
    $LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a \
    -o kernel.elf
```

**Libraries linked:**
| Library | Contents |
|---------|----------|
| `libvortex.a` | Vortex runtime: `vx_spawn_threads`, `vx_barrier`, `vx_printf`, startup code |
| `libc` | Standard C library (malloc, memcpy, etc.) |
| `libm` | Math library (sin, cos, sqrt, etc.) |
| `libclang_rt.builtins` | Compiler builtins (__muldi3, __divsi3, etc.) |

### Step 4: Generate Binary Formats

```bash
# Disassembly for debugging
$LLVM_VORTEX/bin/llvm-objdump -D kernel.elf > kernel.dump

# Raw binary
$LLVM_VORTEX/bin/llvm-objcopy -O binary kernel.elf kernel.bin

# Vortex binary format (with VMA metadata)
OBJCOPY=$LLVM_VORTEX/bin/llvm-objcopy \
    python3 $VORTEX_HOME/kernel/scripts/vxbin.py kernel.elf kernel.vxbin
```

## Complete Script

```bash
#!/bin/bash
# compile_hip_kernel.sh - Compile HIP kernel to Vortex binary

set -e

# Configuration
VORTEX_HIP=/path/to/vortex_hip
VORTEX_HOME=$VORTEX_HIP/vortex
LLVM_VORTEX=/path/to/llvm-vortex
POLYGEIST=$VORTEX_HIP/Polygeist/build
LIBC_VORTEX=/path/to/libc32
LIBCRT_VORTEX=/path/to/libcrt32

INPUT=$1
BASENAME=$(basename "$INPUT" .hip)
OUTDIR=${2:-.}

echo "=== Compiling $INPUT to Vortex binary ==="

# Step 1: HIP -> MLIR (GPU dialect)
echo "[1/5] HIP -> GPU MLIR"
$POLYGEIST/bin/cgeist \
    --cuda-path=/usr/local/cuda \
    --cuda-gpu-arch=sm_70 \
    -I/usr/local/cuda/include \
    "$INPUT" \
    -O2 --function='*' -S --memref-fullrank \
    -o "$OUTDIR/$BASENAME.gpu.mlir"

# Step 2: GPU MLIR -> LLVM MLIR (with Vortex lowering)
echo "[2/5] GPU MLIR -> LLVM MLIR (Vortex lowering)"
$POLYGEIST/bin/polygeist-opt \
    --convert-gpu-to-vortex \
    --gpu-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --convert-cf-to-llvm \
    --convert-scf-to-cf \
    --reconcile-unrealized-casts \
    "$OUTDIR/$BASENAME.gpu.mlir" \
    -o "$OUTDIR/$BASENAME.llvm.mlir"

# Step 3: LLVM MLIR -> LLVM IR
echo "[3/5] LLVM MLIR -> LLVM IR"
$POLYGEIST/bin/mlir-translate \
    --mlir-to-llvmir \
    "$OUTDIR/$BASENAME.llvm.mlir" \
    -o "$OUTDIR/$BASENAME.ll"

# Step 4: LLVM IR -> Object file
echo "[4/5] LLVM IR -> RV32 Object"
$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -mcmodel=medany \
    -O3 \
    -fno-rtti -fno-exceptions \
    -fdata-sections -ffunction-sections \
    -Xclang -target-feature -Xclang +vortex \
    -Xclang -target-feature -Xclang +zicond \
    -mllvm -disable-loop-idiom-all \
    -I$VORTEX_HOME/kernel/include \
    -c "$OUTDIR/$BASENAME.ll" \
    -o "$OUTDIR/$BASENAME.o"

# Step 5: Link to ELF
echo "[5/5] Linking -> ELF"
$LLVM_VORTEX/bin/clang \
    -target riscv32-unknown-elf \
    -march=rv32imaf -mabi=ilp32f \
    -nostartfiles -nostdlib \
    -Wl,-Bstatic,--gc-sections \
    -Wl,-T,$VORTEX_HOME/kernel/scripts/link32.ld \
    -Wl,--defsym=STARTUP_ADDR=0x80000000 \
    "$OUTDIR/$BASENAME.o" \
    $VORTEX_HOME/build/kernel/libvortex.a \
    -L$LIBC_VORTEX/lib -lm -lc \
    $LIBCRT_VORTEX/lib/baremetal/libclang_rt.builtins-riscv32.a \
    -o "$OUTDIR/$BASENAME.elf"

# Generate auxiliary outputs
$LLVM_VORTEX/bin/llvm-objdump -D "$OUTDIR/$BASENAME.elf" > "$OUTDIR/$BASENAME.dump"
$LLVM_VORTEX/bin/llvm-objcopy -O binary "$OUTDIR/$BASENAME.elf" "$OUTDIR/$BASENAME.bin"
OBJCOPY=$LLVM_VORTEX/bin/llvm-objcopy \
    python3 $VORTEX_HOME/kernel/scripts/vxbin.py \
    "$OUTDIR/$BASENAME.elf" "$OUTDIR/$BASENAME.vxbin"

echo "=== Complete ==="
echo "Output files:"
echo "  $OUTDIR/$BASENAME.elf    - ELF executable"
echo "  $OUTDIR/$BASENAME.vxbin  - Vortex loadable binary"
echo "  $OUTDIR/$BASENAME.dump   - Disassembly"
```

## Memory Layout

The linker script (`link32.ld`) places the kernel at:

```
Address         Section
────────────────────────
0x80000000      .text (entry: _start)
                .rodata
                .data
                .bss
                (heap grows up)
                ...
                (stack grows down)
0xFFFFFFFF      Top of memory
```

## Runtime Loading

The host application loads and executes the kernel:

```c
#include <vortex.h>

int main() {
    vx_device_h device;
    vx_buffer_h kernel_buf, args_buf;

    // Open device
    vx_dev_open(&device);

    // Upload kernel binary
    vx_upload_kernel_file(device, "kernel.vxbin", &kernel_buf);

    // Allocate and fill argument buffer
    vx_mem_alloc(device, sizeof(kernel_args_t), VX_MEM_READ, &args_buf);
    kernel_args_t args = { ... };
    vx_copy_to_dev(args_buf, &args, sizeof(args), 0);

    // Execute kernel
    vx_start(device, kernel_buf, args_buf);
    vx_ready_wait(device, VX_MAX_TIMEOUT);

    // Cleanup
    vx_buf_free(args_buf);
    vx_buf_free(kernel_buf);
    vx_dev_close(device);
}
```

## Required External Symbols

The lowered LLVM IR will contain external references that must be provided by libvortex.a:

| Symbol | Provided By | Purpose |
|--------|-------------|---------|
| `vx_get_threadIdx` | vx_spawn.h (inline) | Get thread index TLS pointer |
| `vx_get_blockIdx` | vx_spawn.h (inline) | Get block index TLS pointer |
| `vx_get_blockDim` | vx_spawn.h (inline) | Get block dimensions pointer |
| `vx_get_gridDim` | vx_spawn.h (inline) | Get grid dimensions pointer |
| `vx_barrier` | vx_intrinsics.h (inline) | Thread synchronization |
| `vx_num_warps` | vx_intrinsics.h (inline) | Get warp count |
| `vx_printf` | vx_print.c | Kernel debug output |
| `threadIdx`, `blockIdx` | vx_spawn.c | TLS dim3 variables |
| `gridDim`, `blockDim` | vx_spawn.c | Global dim3 variables |

## Troubleshooting

### Undefined symbol errors

If linking fails with undefined symbols like `vx_get_threadIdx`:
1. Ensure `-I$VORTEX_HOME/kernel/include` is passed during compilation
2. The accessor functions are `static inline` in headers - they should be inlined

### Relocation errors

If the linker reports relocation errors:
1. Check `-mcmodel=medany` is set
2. Verify `STARTUP_ADDR` matches expected load address
3. Ensure all code fits within addressable range

### Missing CSR instructions

If Vortex CSR instructions aren't recognized:
1. Verify `-Xclang -target-feature -Xclang +vortex` is passed
2. Check llvm-vortex is the correct version with Vortex support

## Next Steps

1. **Implement host-side kernel launch wrapper** - Generate host code that calls `vx_spawn_threads` with correct arguments
2. **Memory allocation translation** - Lower `hipMalloc`/`hipMemcpy` to Vortex equivalents
3. **Shared memory support** - Lower `gpu.memref.global` with address space 3 to Vortex local memory
4. **Multi-kernel support** - Handle multiple kernels in single compilation unit
