# Vortex Runtime API Reference

Documentation of the Vortex runtime API used by hip_vortex_runtime.

**Header**: `vortex/runtime/include/vortex.h`
**Library**: `vortex/build/runtime/libvortex.so`

---

## Overview

The Vortex runtime API provides low-level control of the Vortex RISC-V GPU accelerator. It uses opaque handle types for devices and memory buffers.

### Handle Types

```c
typedef struct vx_device* vx_device_h;    // Device handle
typedef struct vx_buffer* vx_buffer_h;    // Memory buffer handle
```

---

## Device Management

### vx_dev_open
```c
int vx_dev_open(vx_device_h* hdevice);
```
**Description**: Open and initialize a Vortex device.

**Parameters**:
- `hdevice`: Output parameter for device handle

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
vx_device_h device;
if (vx_dev_open(&device) != 0) {
    // Handle error
}
```

---

### vx_dev_close
```c
int vx_dev_close(vx_device_h hdevice);
```
**Description**: Close and release a Vortex device.

**Parameters**:
- `hdevice`: Device handle to close

**Returns**: 0 on success, non-zero on error

---

### vx_dev_caps
```c
int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t* value);
```
**Description**: Query device capabilities.

**Parameters**:
- `hdevice`: Device handle
- `caps_id`: Capability identifier (see VX_CAPS_* constants)
- `value`: Output parameter for capability value

**Capability IDs**:
- `VX_CAPS_VERSION`: Device version
- `VX_CAPS_NUM_THREADS`: Threads per warp
- `VX_CAPS_NUM_WARPS`: Warps per core
- `VX_CAPS_NUM_CORES`: Total cores
- `VX_CAPS_CACHE_LINE_SIZE`: Cache line size in bytes
- `VX_CAPS_LOCAL_MEM_SIZE`: Shared/local memory size per core
- `VX_CAPS_ISA_FLAGS`: Supported RISC-V ISA extensions

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
uint64_t num_cores;
vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores);
```

---

## Memory Management

### vx_mem_alloc
```c
int vx_mem_alloc(vx_device_h hdevice, uint64_t size, int flags, vx_buffer_h* hbuffer);
```
**Description**: Allocate device memory.

**Parameters**:
- `hdevice`: Device handle
- `size`: Size in bytes to allocate
- `flags`: Allocation flags (0 for default)
- `hbuffer`: Output parameter for buffer handle

**Returns**: 0 on success, non-zero on error

**Notes**:
- Buffer handle is opaque; cannot be dereferenced directly
- Use vx_copy_to_dev/vx_copy_from_dev to access memory
- Must call vx_mem_free when done

**Usage**:
```c
vx_buffer_h buffer;
if (vx_mem_alloc(device, 1024, 0, &buffer) != 0) {
    // Handle error
}
```

---

### vx_mem_free
```c
int vx_mem_free(vx_buffer_h hbuffer);
```
**Description**: Free device memory.

**Parameters**:
- `hbuffer`: Buffer handle to free

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
vx_mem_free(buffer);
```

---

### vx_buf_access
```c
int vx_buf_access(vx_buffer_h hbuffer, uint64_t offset, uint64_t size, int flags);
```
**Description**: Set memory access permissions for a buffer region.

**Parameters**:
- `hbuffer`: Buffer handle
- `offset`: Offset into buffer in bytes
- `size`: Size of region in bytes
- `flags`: Access flags (read/write permissions)

**Returns**: 0 on success, non-zero on error

---

## Data Transfer

### vx_copy_to_dev
```c
int vx_copy_to_dev(vx_buffer_h hbuffer, const void* host_ptr, uint64_t dst_offset, uint64_t size);
```
**Description**: Copy data from host to device memory.

**Parameters**:
- `hbuffer`: Destination buffer handle
- `host_ptr`: Source pointer in host memory
- `dst_offset`: Offset into device buffer in bytes
- `size`: Number of bytes to copy

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
float data[256];
// Initialize data...
vx_copy_to_dev(buffer, data, 0, sizeof(data));
```

---

### vx_copy_from_dev
```c
int vx_copy_from_dev(void* host_ptr, vx_buffer_h hbuffer, uint64_t src_offset, uint64_t size);
```
**Description**: Copy data from device to host memory.

**Parameters**:
- `host_ptr`: Destination pointer in host memory
- `hbuffer`: Source buffer handle
- `src_offset`: Offset into device buffer in bytes
- `size`: Number of bytes to copy

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
float result[256];
vx_copy_from_dev(result, buffer, 0, sizeof(result));
```

---

## Kernel Execution

### vx_upload_kernel_bytes
```c
int vx_upload_kernel_bytes(vx_device_h hdevice, const void* content, uint64_t size);
```
**Description**: Upload kernel binary to device.

**Parameters**:
- `hdevice`: Device handle
- `content`: Pointer to kernel binary data
- `size`: Size of kernel binary in bytes

**Returns**: 0 on success, non-zero on error

**Notes**:
- Kernel must be compiled for RISC-V target
- Binary format depends on Vortex build configuration

**Usage**:
```c
// Load kernel binary from file or embedded data
const uint8_t kernel_binary[] = { /* ... */ };
vx_upload_kernel_bytes(device, kernel_binary, sizeof(kernel_binary));
```

---

### vx_upload_kernel_file
```c
int vx_upload_kernel_file(vx_device_h hdevice, const char* filename);
```
**Description**: Upload kernel binary from file.

**Parameters**:
- `hdevice`: Device handle
- `filename`: Path to kernel binary file

**Returns**: 0 on success, non-zero on error

**Usage**:
```c
vx_upload_kernel_file(device, "vecadd_kernel.riscv");
```

---

### vx_start
```c
int vx_start(vx_device_h hdevice);
```
**Description**: Start kernel execution on device.

**Parameters**:
- `hdevice`: Device handle

**Returns**: 0 on success, non-zero on error

**Notes**:
- Kernel must be uploaded first
- Kernel arguments must be set (implementation-specific)
- Non-blocking; use vx_ready_wait for synchronization

---

### vx_ready_wait
```c
int vx_ready_wait(vx_device_h hdevice, int64_t timeout);
```
**Description**: Wait for device to complete execution.

**Parameters**:
- `hdevice`: Device handle
- `timeout`: Timeout in nanoseconds, or special values:
  - `-1` (VX_MAX_TIMEOUT): Wait indefinitely
  - `0`: Poll without waiting

**Returns**: 0 on success, non-zero on timeout/error

**Usage**:
```c
// Start kernel
vx_start(device);

// Wait for completion (infinite timeout)
if (vx_ready_wait(device, VX_MAX_TIMEOUT) != 0) {
    // Execution failed or timed out
}
```

---

## Constants

### Timeout Values
```c
#define VX_MAX_TIMEOUT  (-1)  // Infinite timeout
```

### Device Capabilities (VX_CAPS_*)
Defined in `VX_config.h` (generated during Vortex build):
- `VX_CAPS_VERSION`
- `VX_CAPS_NUM_THREADS`
- `VX_CAPS_NUM_WARPS`
- `VX_CAPS_NUM_CORES`
- `VX_CAPS_CACHE_LINE_SIZE`
- `VX_CAPS_LOCAL_MEM_SIZE`
- `VX_CAPS_GLOBAL_MEM_SIZE`
- `VX_CAPS_ISA_FLAGS`

---

## Typical Usage Pattern

```c
#include <vortex.h>

int main() {
    vx_device_h device;
    vx_buffer_h input_buf, output_buf;

    // 1. Open device
    vx_dev_open(&device);

    // 2. Allocate device memory
    vx_mem_alloc(device, INPUT_SIZE, 0, &input_buf);
    vx_mem_alloc(device, OUTPUT_SIZE, 0, &output_buf);

    // 3. Copy data to device
    vx_copy_to_dev(input_buf, host_input, 0, INPUT_SIZE);

    // 4. Upload kernel
    vx_upload_kernel_file(device, "kernel.riscv");

    // 5. Start execution
    vx_start(device);

    // 6. Wait for completion
    vx_ready_wait(device, VX_MAX_TIMEOUT);

    // 7. Copy results back
    vx_copy_from_dev(host_output, output_buf, 0, OUTPUT_SIZE);

    // 8. Cleanup
    vx_mem_free(input_buf);
    vx_mem_free(output_buf);
    vx_dev_close(device);

    return 0;
}
```

---

## HIP to Vortex API Mapping

### Memory Management
| HIP API | Vortex API |
|---------|------------|
| `hipMalloc(ptr, size)` | `vx_mem_alloc(device, size, 0, &buffer)` |
| `hipFree(ptr)` | `vx_mem_free(buffer)` |
| `hipMemcpy(dst, src, size, H2D)` | `vx_copy_to_dev(buffer, src, 0, size)` |
| `hipMemcpy(dst, src, size, D2H)` | `vx_copy_from_dev(dst, buffer, 0, size)` |

### Device Management
| HIP API | Vortex API |
|---------|------------|
| `hipSetDevice(id)` | `vx_dev_open(&device)` |
| `hipDeviceSynchronize()` | `vx_ready_wait(device, VX_MAX_TIMEOUT)` |
| `hipDeviceReset()` | `vx_dev_close(device)` |
| `hipGetDeviceProperties(prop, id)` | `vx_dev_caps(device, VX_CAPS_*, &value)` |

### Kernel Execution
| HIP API | Vortex API |
|---------|------------|
| `kernel<<<grid, block>>>(args)` | `vx_upload_kernel_file()` + `vx_start()` |
| (implicit synchronization) | `vx_ready_wait(device, timeout)` |

---

## Notes

1. **Buffer Handles vs Pointers**: Vortex uses opaque buffer handles (`vx_buffer_h`), not raw pointers. Host code cannot directly dereference device memory addresses.

2. **Synchronous Operations**: Most Vortex API calls are synchronous except `vx_start()`. Always call `vx_ready_wait()` after `vx_start()`.

3. **Error Handling**: All functions return `int` status (0 = success). Check return values for error handling.

4. **Single Device**: Current implementation assumes single-device model. Multi-device support may require additional state management.

5. **Kernel Arguments**: The Vortex API for passing kernel arguments is implementation-specific and may vary by Vortex build configuration. Typically involves writing argument struct to a specific memory region or register.

---

## References

- **Vortex Repository**: [https://github.com/vortexgpgpu/vortex](https://github.com/vortexgpgpu/vortex)
- **Vortex Documentation**: See `vortex/docs/`
- **Header File**: `vortex/runtime/include/vortex.h`
- **Example Code**: `vortex/tests/runtime/`
