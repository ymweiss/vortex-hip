# HIP Implementation Comparison Reference

## Quick Comparison Table

| Feature | chipStar | HIP-CPU | Vortex (Target) |
|---------|----------|---------|-----------------|
| **Purpose** | SPIR-V GPU runtime | CPU fallback | RISC-V GPU runtime |
| **Target Hardware** | OpenCL/Level0 GPUs | CPU cores | Vortex RISC-V GPU |
| **Lines of Code** | ~50,000 | ~2,000 | ~10,000-15,000 (est.) |
| **Compilation** | Clang → SPIR-V | Direct C++ | Clang → SPIR-V/RISC-V |
| **JIT Compilation** | Yes (driver) | No | Likely yes |
| **Memory Model** | Discrete (Host ≠ Device) | Unified (Host = Device) | TBD |
| **Thread Creation** | Backend-managed | 9-17 OS threads | Hardware threads |
| **Barrier Implementation** | Hardware | O(n) fiber switches | Hardware (expected) |
| **Shared Memory** | On-chip SRAM | `thread_local` | On-chip (expected) |
| **Performance vs Native GPU** | 1x (native) | 0.01-0.1x | 1x (goal) |

---

## API Implementation Comparison

### hipMalloc

**chipStar (50+ lines):**
```cpp
hipError_t hipMalloc(void** ptr, size_t size) {
    // 1. Get active device
    // 2. Get active context
    // 3. Call backend allocate
    // 4. Track allocation
    // 5. Update statistics
    // 6. Error handling
    auto device = Backend->getActiveDevice();
    auto context = device->getContext();
    *ptr = context->allocate(size, alignment, memoryType);
    track_allocation(*ptr, size);
    return hipSuccess;
}
```

**HIP-CPU (8 lines):**
```cpp
hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) return hipErrorInvalidValue;
    *ptr = std::malloc(size);
    if (!ptr) return hipErrorOutOfMemory;
    return hipSuccess;
}
```

**Vortex (recommended - 15 lines):**
```cpp
hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) return hipErrorInvalidValue;
    if (size == 0) return hipSuccess;

    uint64_t dev_addr;
    int result = vortex_mem_alloc(g_device, size, &dev_addr);
    if (result != 0) return hipErrorOutOfMemory;

    *ptr = reinterpret_cast<void*>(dev_addr);
    track_allocation(*ptr, size);  // For debugging

    return hipSuccess;
}
```

---

### hipMemcpy

**chipStar (100+ lines):**
```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    // 1. Validate parameters
    // 2. Determine memory types (host/device)
    // 3. Select transfer path
    // 4. Wait for device idle (synchronous)
    // 5. Perform transfer via backend API
    // 6. Wait for completion
    // 7. Error handling
    synchronize_device();

    switch (kind) {
    case hipMemcpyHostToDevice:
        backend->copyToDevice(dst, src, size);
        break;
    // ... other cases
    }

    synchronize_device();
    return hipSuccess;
}
```

**HIP-CPU (10 lines):**
```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (size == 0) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;

    synchronize_device();
    std::memcpy(dst, src, size);  // All memory is CPU RAM

    return hipSuccess;
}
```

**Vortex (recommended - 30 lines):**
```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (size == 0) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;

    // Wait for pending operations
    vortex_wait(g_device);

    int result;
    switch (kind) {
    case hipMemcpyHostToDevice:
        result = vortex_copy_to_dev(g_device, (uint64_t)dst, src, size);
        break;
    case hipMemcpyDeviceToHost:
        result = vortex_copy_from_dev(g_device, dst, (uint64_t)src, size);
        break;
    case hipMemcpyDeviceToDevice:
        result = vortex_copy_dev_to_dev(g_device, (uint64_t)dst, (uint64_t)src, size);
        break;
    case hipMemcpyHostToHost:
        memcpy(dst, src, size);
        result = 0;
        break;
    default:
        return hipErrorInvalidValue;
    }

    return (result == 0) ? hipSuccess : hipErrorUnknown;
}
```

---

### hipLaunchKernel

**chipStar (200+ lines across multiple functions):**
```cpp
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream) {
    // 1. Find kernel by host function pointer
    auto device = Backend->getActiveDevice();
    auto kernel = device->findKernel(func);
    if (!kernel) compile_and_register_kernel(func);

    // 2. Get function metadata (argument types)
    auto func_info = kernel->getFuncInfo();

    // 3. Create execution item
    auto exec_item = backend->createExecItem(grid, block, sharedMem, stream);
    exec_item->setKernel(kernel);
    exec_item->setArgs(args);

    // 4. Marshal arguments based on metadata
    if (func_info->hasByRefArgs()) {
        // Allocate spill buffer for large arguments
        allocate_arg_spill_buffer(exec_item, func_info);
    }

    // Visit each argument and setup
    func_info->visitKernelArgs(args, [&](const Arg& arg) {
        switch (arg.kind) {
        case POD:
            backend->setKernelArg(kernel, arg.index, arg.size, arg.data);
            break;
        case Pointer:
            backend->setKernelArgPointer(kernel, arg.index, arg.data);
            break;
        // ... other types
        }
    });

    // 5. Submit to backend queue
    stream->launch(exec_item);

    return hipSuccess;
}
```

**HIP-CPU (40 lines):**
```cpp
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream) {
    if (!stream) stream = default_stream();

    // Enqueue lambda that executes the kernel
    stream->enqueue([=]() {
        // Parallel for each block
        std::for_each(std::execution::par_unseq,
                     block_iterator(0),
                     block_iterator(grid.x * grid.y * grid.z),
                     [&](int block_id) {
            // Set block ID
            set_block_id(block_id);

            // Execute all threads in block
            if (!uses_barriers) {
                // Fast path: simple loop
                for (int tid = 0; tid < block_size; tid++) {
                    set_thread_id(tid);
                    kernel_function(args...);
                }
            } else {
                // Slow path: fiber-based
                execute_with_fibers(kernel_function, args, block_size);
            }
        });
    });

    return hipSuccess;
}
```

**Vortex (recommended - 60 lines):**
```cpp
hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
                           void** args, size_t sharedMem, hipStream_t stream) {
    // 1. Find kernel
    auto kernel_info = g_registry.findKernel(func);
    if (!kernel_info) return hipErrorInvalidDeviceFunction;

    // 2. Marshal arguments to buffer
    std::vector<uint8_t> arg_buffer;
    for (size_t i = 0; i < kernel_info->num_args; i++) {
        const auto& arg_meta = kernel_info->arg_types[i];
        void* arg_ptr = args[i];

        if (arg_meta.is_pointer) {
            // Pass device address (8 bytes)
            uint64_t addr = reinterpret_cast<uint64_t>(arg_ptr);
            append_to_buffer(arg_buffer, &addr, sizeof(addr));
        } else {
            // Pass value directly
            append_to_buffer(arg_buffer, arg_ptr, arg_meta.size);
        }
    }

    // 3. Setup kernel configuration
    vortex_kernel_config_t config;
    config.grid_dim[0] = grid.x;
    config.grid_dim[1] = grid.y;
    config.grid_dim[2] = grid.z;
    config.block_dim[0] = block.x;
    config.block_dim[1] = block.y;
    config.block_dim[2] = block.z;
    config.shared_mem_size = sharedMem;

    // 4. Submit to Vortex
    int result = vortex_kernel_enqueue(
        g_device,
        stream ? stream->handle : g_default_stream,
        kernel_info->device_code,
        &config,
        arg_buffer.data(),
        arg_buffer.size()
    );

    return (result == 0) ? hipSuccess : hipErrorLaunchFailure;
}
```

---

## Kernel Registration

### chipStar Approach

```cpp
// Compiler-generated code (automatic)
__attribute__((constructor))
void __chipstar_register_module() {
    void** handle = __hipRegisterFatBinary(&__hip_fatbin_data);
    __hipRegisterFunction(handle, (void*)&myKernel, "_Z8myKernelPfi");
}

// Runtime implementation
void** __hipRegisterFatBinary(const void* data) {
    // Extract SPIR-V from fat binary
    auto spirv = extractSPIRVModule(data);

    // Register with global registry
    auto handle = SPVRegister::get().registerSource(spirv);

    return handle;
}

void __hipRegisterFunction(void** handle, const void* func_ptr, const char* name) {
    SPVRegister::get().bindFunction(handle, func_ptr, name);
}
```

### Vortex Recommendation

```cpp
// Same compiler-generated pattern
__attribute__((constructor))
void __vortex_register_module() {
    void** handle = __hipRegisterFatBinary(&__hip_fatbin_data);
    __hipRegisterFunction(handle, (void*)&myKernel, "_Z8myKernelPfi");
}

// Vortex runtime implementation
void** __hipRegisterFatBinary(const void* data) {
    // Extract device binary (SPIR-V or Vortex binary)
    auto binary = extractDeviceBinary(data);

    // Load into Vortex memory
    void* device_code = vortex_load_kernel(g_device, binary.data(), binary.size());

    // Register module
    auto handle = VortexRegistry::get().registerModule(device_code);

    return handle;
}

void __hipRegisterFunction(void** handle, const void* func_ptr, const char* name) {
    VortexRegistry::get().bindKernel(handle, func_ptr, name);
}
```

---

## Memory Architecture Comparison

### chipStar (Discrete Memory)

```
Host Side:
  - CPU RAM (malloc/free)
  - Host pointers (0x7f...)

Device Side:
  - GPU VRAM (driver alloc)
  - Device pointers (0x...)
  - Separate address space

Transfer:
  - PCI-e DMA (async)
  - Bandwidth: 16-32 GB/s
  - Latency: 10-1000 μs
```

### HIP-CPU (Unified Memory)

```
Everything in CPU RAM:
  - malloc() for everything
  - All pointers are CPU pointers
  - No transfers needed

"Copy":
  - Just memcpy()
  - Bandwidth: 50-200 GB/s
  - Latency: ~0 μs
```

### Vortex Options

**Option A: Unified (Recommended if hardware supports)**
```
Shared Address Space:
  - Single allocator
  - All pointers valid everywhere
  - Explicit flush/invalidate for caches

Advantages:
  - Simplest runtime
  - Zero-copy possible
  - Easy debugging
```

**Option B: Discrete**
```
Separate Address Spaces:
  - Host allocator (malloc)
  - Device allocator (vortex_alloc)
  - Explicit transfers

Advantages:
  - More control
  - Better isolation
  - Standard GPU model
```

---

## Thread/Fiber Model

### chipStar (Hardware Threads)

```
1 HIP Thread = 1 Hardware Wavefront Lane
  - Thousands of hardware threads
  - Hardware barriers (free)
  - Hardware shared memory
  - True parallelism

Example:
  kernel<<<512, 256>>>()
  = 131,072 HIP threads
  = Runs on ~64 compute units
  = True parallel execution
```

### HIP-CPU (Software Fibers)

```
1 HIP Thread = 1 Fiber (if barriers) or 1 Loop Iteration (if no barriers)
  - 8-16 OS threads
  - Software barriers (expensive)
  - thread_local "shared" memory
  - Simulated parallelism

Example:
  kernel<<<512, 256>>>()
  = 131,072 HIP threads
  = 8-16 OS threads
  = 131,072 / 16 = 8,192:1 mapping
```

### Vortex (Expected: Hardware Threads)

```
1 HIP Thread = 1 RISC-V Hart in Wavefront
  - Hardware threads (warps/wavefronts)
  - Hardware barriers (expected)
  - Hardware shared memory (expected)
  - True parallelism

Configuration to determine:
  - How many threads per block?
  - How many blocks in parallel?
  - Warp/wavefront size?
  - Barrier synchronization scope?
```

---

## Feature Support Matrix

| Feature | chipStar | HIP-CPU | Vortex Goal |
|---------|----------|---------|-------------|
| **Basic Kernels** | ✅ Full | ✅ Full | ✅ Target |
| **threadIdx/blockIdx** | ✅ Full | ✅ Full | ✅ Target |
| **Shared Memory** | ✅ Fast | ⚠️ Slow (thread_local) | ✅ Target |
| **__syncthreads()** | ✅ Free (hardware) | ⚠️ Expensive (O(n)) | ✅ Target |
| **Atomics** | ✅ Full | ✅ Via std::atomic | ✅ Target |
| **Textures** | ✅ Full | ❌ Not supported | ⚠️ Optional |
| **Streams** | ✅ Full async | ✅ Basic | ✅ Target |
| **Events** | ✅ Full | ✅ Basic | ✅ Target |
| **Multi-GPU** | ✅ Yes | ❌ No | ⚠️ Future |
| **Dynamic Parallelism** | ❌ No | ❌ No | ❌ Not planned |
| **Warp Primitives** | ✅ Full | ⚠️ Limited | ✅ Target |

Legend:
- ✅ Fully supported
- ⚠️ Limited/slow implementation
- ❌ Not supported

---

## Performance Characteristics

### Memory Bandwidth

| Implementation | Host→Device | Device→Host | Device→Device |
|----------------|-------------|-------------|---------------|
| chipStar (PCIe 4.0) | 32 GB/s | 32 GB/s | 500-2000 GB/s |
| HIP-CPU | N/A (unified) | N/A (unified) | 50-200 GB/s |
| Vortex (estimate) | TBD | TBD | TBD |

### Kernel Launch Overhead

| Implementation | Launch Time | Notes |
|----------------|-------------|-------|
| chipStar | 1-10 μs | Cached kernels, backend submission |
| HIP-CPU | 1-5 μs | C++ function call overhead |
| Vortex (target) | < 10 μs | Goal for competitive performance |

### Barrier Cost

| Implementation | Cost per __syncthreads() | Notes |
|----------------|--------------------------|-------|
| chipStar | ~0 ns | Hardware barrier |
| HIP-CPU | blockDim × 50 ns | Fiber context switches |
| Vortex (target) | ~0 ns | Hardware barrier expected |

---

## Code Size Comparison

### Complete Implementation

| Component | chipStar | HIP-CPU | Vortex (est.) |
|-----------|----------|---------|---------------|
| Device API | ~6,000 | ~500 | ~2,000 |
| Memory Management | ~3,000 | ~200 | ~1,000 |
| Kernel Launch | ~2,000 | ~300 | ~1,500 |
| Backend | ~10,000 | N/A | ~3,000 |
| SPIR-V/Loader | ~5,000 | N/A | ~2,000 |
| Utilities | ~5,000 | ~500 | ~1,000 |
| **Total** | **~30,000** | **~1,500** | **~10,500** |

### Individual API Functions

| Function | chipStar | HIP-CPU | Vortex (est.) |
|----------|----------|---------|---------------|
| hipMalloc | 50 | 8 | 15 |
| hipFree | 30 | 6 | 10 |
| hipMemcpy | 100 | 10 | 30 |
| hipLaunchKernel | 200 | 40 | 60 |
| hipDeviceSynchronize | 50 | 15 | 20 |
| **Average Complexity** | **High** | **Low** | **Medium** |

---

## Recommended Vortex Approach

### Starting Point

Use **HIP-CPU's simplicity** for:
- Initial API structure
- Error handling patterns
- Test framework

Adopt **chipStar's sophistication** for:
- Kernel registry (SPVRegister pattern)
- Argument marshalling (metadata-driven)
- Backend abstraction
- SPIR-V loading (if applicable)

### Hybrid Architecture

```
┌─────────────────────────────────────────┐
│   HIP API Layer (HIP-CPU inspired)      │
│   - Simple, direct implementations      │
│   - ~2,000 lines                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Kernel Management (chipStar inspired) │
│   - SPVRegister pattern                 │
│   - Metadata-driven arguments           │
│   - ~3,000 lines                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Vortex Backend (Custom)               │
│   - Device initialization               │
│   - Memory management                   │
│   - Kernel execution                    │
│   - ~5,000 lines                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Vortex Driver/Hardware                │
└─────────────────────────────────────────┘
```

**Total Estimated Size:** ~10,000-15,000 lines

---

## Quick Decision Guide

### When to follow chipStar:

✅ Kernel registration mechanism
✅ Argument metadata extraction
✅ Backend abstraction pattern
✅ SPIR-V loading (if using SPIR-V)
✅ Module compilation/caching
✅ Stream/event management

### When to follow HIP-CPU:

✅ Initial API prototyping
✅ Simple memory management
✅ Direct function implementations
✅ Testing framework structure
✅ Error code patterns
✅ Minimal viable product

### Vortex-specific decisions:

🎯 Memory model (unified vs discrete)
🎯 Compilation pipeline (SPIR-V vs direct)
🎯 Driver interface design
🎯 Thread mapping strategy
🎯 Shared memory implementation
🎯 Performance optimization focus

---

## File Organization

### Recommended Structure

```
vortex_hip/
├── include/
│   └── hip/
│       ├── hip_runtime.h          (Public API - from HIP headers)
│       └── hip_runtime_api.h      (API declarations)
│
├── src/
│   ├── runtime/
│   │   ├── device.cpp             (Device management - 500 lines)
│   │   ├── memory.cpp             (Memory ops - 1000 lines)
│   │   ├── stream.cpp             (Stream/event - 800 lines)
│   │   └── error.cpp              (Error handling - 200 lines)
│   │
│   ├── kernel/
│   │   ├── registry.cpp           (Kernel registry - 800 lines, chipStar-inspired)
│   │   ├── launch.cpp             (Launch logic - 1500 lines)
│   │   ├── arguments.cpp          (Arg marshalling - 600 lines)
│   │   └── metadata.cpp           (Metadata parsing - 500 lines)
│   │
│   ├── backend/
│   │   ├── vortex_backend.cpp     (Backend impl - 2000 lines)
│   │   ├── vortex_driver.cpp      (Driver interface - 1000 lines)
│   │   └── vortex_loader.cpp      (Binary loading - 800 lines)
│   │
│   └── bindings.cpp               (Registration stubs - 400 lines)
│
├── tests/
│   ├── unit/                      (Unit tests per component)
│   ├── integration/               (Kernel tests)
│   └── benchmarks/                (Performance tests)
│
└── tools/
    ├── vortex-hipcc               (Compiler wrapper)
    └── vortex-spirv-trans         (SPIR-V translator, if needed)
```

**Total:** ~10,100 lines (organized and maintainable)

---

## Summary: Key Takeaways

| Aspect | Recommendation |
|--------|----------------|
| **Starting Point** | HIP-CPU for rapid prototyping |
| **Architecture** | chipStar patterns for production |
| **Complexity** | 10,000-15,000 lines (medium) |
| **Timeline** | 3 months for MVP, 6 months for full |
| **Critical Path** | Hardware barrier support |
| **Memory Model** | Unified if possible, else discrete |
| **Compilation** | SPIR-V preferred, LLVM fallback |

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Purpose:** Quick reference for implementation decisions
**Audience:** Vortex HIP implementation team
