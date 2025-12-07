# GPU to Vortex MLIR Lowering Reference

**Purpose:** Companion document to WORK_DISTRIBUTION.md providing a concise guide for implementing GPU dialect lowering in the GPUToVortex pass.

**Target Architecture:** RV32 (32-bit RISC-V)

---

## GPU Dialect Operations

### Operations Requiring Custom Lowering

| Operation | Vortex Lowering |
|-----------|-----------------|
| `gpu.module` | Convert to LLVM module with Vortex kernel markers |
| `gpu.func` | Lower to LLVM function with kernel attribute |
| `gpu.block_id {x\|y\|z}` | CSR read: `csrr blockIdx.{x\|y\|z}` |
| `gpu.thread_id {x\|y\|z}` | CSR read: `csrr threadIdx.{x\|y\|z}` |
| `gpu.barrier` | `vx_barrier(barrier_id, num_warps)` - see below |
| `gpu.return` | Standard ret (handled by gpu-to-llvm) |

### Barrier Lowering Details

```c
// Vortex signature (from vx_intrinsics.h)
void vx_barrier(int barrier_id, int num_warps);
```

- **barrier_id:** Unique ID for this barrier (use 0 for simple cases, or assign unique IDs per barrier site)
- **num_warps:** Number of warps participating in the barrier

For `gpu.barrier` lowering:
```llvm
; Get number of warps in the block
%num_warps = call i32 @vx_num_warps()
; Issue barrier
call void @vx_barrier(i32 0, i32 %num_warps)
```

---

## Memref Address Spaces

| Address Space | MLIR Syntax | Meaning | Vortex Mapping |
|---------------|-------------|---------|----------------|
| 0 (default) | `memref<?xf32>` | Global memory | Device DRAM |
| 3 | `memref<?xf32, 3>` | Shared memory | Vortex shared memory (per-warp) |

---

## Shared Memory Lowering

### MLIR Pattern

```mlir
memref.global @shared_mem : memref<1xf32, 3> = uninitialized

gpu.func @kernel(...) {
  %shmem = memref.get_global @shared_mem : memref<1xf32, 3>
  memref.store %val, %shmem[%idx] : memref<1xf32, 3>
  %loaded = memref.load %shmem[%idx] : memref<1xf32, 3>
}
```

### Lowering Strategy

1. **`memref.global` (address space 3):** Emit as shared memory allocation in `.shared` section
2. **`memref.get_global`:** Lower to pointer to shared memory base
3. **`memref.load/store` (address space 3):** Generate load/store with shared memory addressing

---

## External Function Calls

| Source | Vortex Lowering |
|--------|-----------------|
| `llvm.call @printf(...)` | Replace with `@vx_printf` |

---

## Vortex Main Wrapper Generation

The `--generate-vortex-main` pass generates the Vortex-specific entry point after gpu-to-llvm lowering:

### Generated Functions

1. **`main()`** - Entry point that:
   - Reads args from `VX_CSR_MSCRATCH` (0x340) via inline assembly
   - Calls `vx_spawn_threads()` with kernel callback

2. **`kernel_body(void* args)`** - Wrapper that:
   - Unpacks arguments from the struct (offset 24 = skip grid_dim[3] + block_dim[3])
   - Calls the original lowered kernel function

### Args Struct Layout

```c
typedef struct {
    uint32_t grid_dim[3];    // bytes 0-11
    uint32_t block_dim[3];   // bytes 12-23
    // User kernel arguments starting at byte 24
} kernel_args_t;
```

### Pass Pipeline

```bash
polygeist-opt input.mlir \
    --convert-gpu-to-vortex \    # 1. Vortex intrinsics + metadata
    --gpu-to-llvm \              # 2. GPU func → LLVM func
    --generate-vortex-main \     # 3. Generate main() + kernel_body
    --convert-func-to-llvm \     # 4. Remaining func ops
    --reconcile-unrealized-casts \
    -o output.mlir
```

### Example Output

```mlir
llvm.func @vx_spawn_threads(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

llvm.func @kernel_body(%arg0: !llvm.ptr) {
  // Unpack args from offset 24
  %0 = llvm.getelementptr %arg0[24] : (!llvm.ptr) -> !llvm.ptr, i8
  %1 = llvm.load %0 : !llvm.ptr -> i32
  // ... load remaining args ...
  llvm.call @_Z13kernel...(%1, ...) : ...
  llvm.return
}

llvm.func @main() -> i32 {
  // Read args from CSR 0x340
  %0 = llvm.inline_asm has_side_effects "csrr $0, 0x340", "=r" : () -> !llvm.struct<(i32)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i32)>
  %2 = llvm.inttoptr %1 : i32 to !llvm.ptr

  // Get block_dim pointer (offset 12)
  %3 = llvm.getelementptr %2[12] : (!llvm.ptr) -> !llvm.ptr, i8

  // Get kernel_body function pointer
  %4 = llvm.mlir.addressof @kernel_body : !llvm.ptr

  // Call vx_spawn_threads(1, grid_dim, block_dim, kernel_body, args)
  %5 = llvm.mlir.constant(1 : i32) : i32
  %6 = llvm.call @vx_spawn_threads(%5, %2, %3, %4, %2) : ... -> i32
  llvm.return %6 : i32
}
```

---

## Test Kernels by Feature

| Feature | Test Kernel |
|---------|-------------|
| Thread/block IDs | vecadd, basic, relu |
| Shared memory | sgemm2 |
| Barriers | sgemm2 |
| Printf | printf_kernel |
| Main wrapper | All kernels (via --generate-vortex-main) |

---

**Last Updated:** 2025-12-05
