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
| `gpu.barrier` | `vx_barrier()` intrinsic |
| `gpu.return` | Standard ret (handled by gpu-to-llvm) |

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

## Test Kernels by Feature

| Feature | Test Kernel |
|---------|-------------|
| Thread/block IDs | vecadd, basic, relu |
| Shared memory | sgemm2 |
| Barriers | sgemm2 |
| Printf | printf_kernel |

---

**Last Updated:** 2025-11-30
