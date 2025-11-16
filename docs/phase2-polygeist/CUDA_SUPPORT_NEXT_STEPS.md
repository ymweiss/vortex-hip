# Polygeist CUDA Support - Next Steps

**Status:** Blocked - Requires Decision
**Date:** 2025-11-15
**Context:** Week 1, Tuesday setup task

---

## Current Situation

We need Polygeist to recognize CUDA/HIP kernel syntax (`__global__`, `<<<>>>`, `threadIdx`, etc.) to convert it to MLIR GPU dialect. However:

**Current Build:** `POLYGEIST_ENABLE_CUDA=0` (CUDA support disabled)
**Issue:** Without CUDA support, Polygeist cannot parse HIP kernel syntax
**Blocker:** Enabling CUDA support requires CUDA toolkit installation

---

## Why CUDA Toolkit is Required

When `POLYGEIST_ENABLE_CUDA=ON` is set, Polygeist's CMake configuration:

1. **lib/polygeist/Passes/CMakeLists.txt:**
   - Calls `find_package(CUDA)` and `enable_language(CUDA)`
   - Requires CUDA headers for gpu-to-cubin pass
   - Links against libcuda.so

2. **lib/polygeist/ExecutionEngine/CMakeLists.txt:**
   - Builds CUDA runtime wrappers
   - Requires CUDA toolkit include directories

3. **tools/cgeist/CMakeLists.txt:**
   - Adds dependency on `execution_engine_cuda_wrapper_binary_include`
   - This target is only created if CUDA toolkit is found

**Result:** Build fails if CUDA toolkit is not installed.

---

## What We Actually Need

We only need **CUDA syntax parsing** (frontend), not **CUDA execution** (backend):

**Need:**
- Recognition of `__global__` attribute
- Parsing of `<<<>>>` kernel launch syntax
- Understanding of `threadIdx`, `blockIdx`, etc.
- Conversion to MLIR GPU dialect operations

**Don't Need:**
- CUDA execution engine wrappers
- GPU-to-CUBIN compilation pass
- CUDA runtime library linkage
- Actual CUDA kernel execution

The parsing logic is in `tools/cgeist/Lib/CGCall.cc` and requires only the `POLYGEIST_ENABLE_CUDA=1` preprocessor definition, not the actual CUDA toolkit.

---

## Options

### Option 1: Install CUDA Toolkit (Easiest)

**Steps:**
```bash
# Install NVIDIA CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Rebuild Polygeist with CUDA support
cd Polygeist/build
cmake .. -DPOLYGEIST_ENABLE_CUDA=ON
ninja
```

**Pros:**
- Simple, standard approach
- No modifications to Polygeist needed
- Fully supported configuration

**Cons:**
- Requires ~3GB download/install
- Adds unnecessary dependency (we don't run CUDA code)
- May conflict with existing GPU drivers

**Time:** ~30 minutes (download + build)

---

### Option 2: Create New CMake Option (Clean)

Add `POLYGEIST_ENABLE_CUDA_SYNTAX_ONLY` option that enables parsing without toolkit.

**Implementation:**

1. **Polygeist/CMakeLists.txt:**
```cmake
set(POLYGEIST_ENABLE_CUDA_SYNTAX_ONLY 0 CACHE BOOL
    "Enable CUDA syntax parsing without requiring CUDA toolkit")

if(POLYGEIST_ENABLE_CUDA_SYNTAX_ONLY)
  set(POLYGEIST_ENABLE_CUDA_FRONTEND 1)
  set(POLYGEIST_SKIP_CUDA_TOOLKIT 1)
endif()
```

2. **tools/cgeist/CMakeLists.txt:**
```cmake
if(POLYGEIST_ENABLE_CUDA OR POLYGEIST_ENABLE_CUDA_FRONTEND)
  target_compile_definitions(cgeist PRIVATE POLYGEIST_ENABLE_CUDA=1)
endif()

if(POLYGEIST_ENABLE_CUDA AND NOT POLYGEIST_SKIP_CUDA_TOOLKIT)
  add_dependencies(cgeist execution_engine_cuda_wrapper_binary_include)
endif()
```

3. **lib/polygeist/Passes/CMakeLists.txt:**
```cmake
if(POLYGEIST_ENABLE_CUDA AND NOT POLYGEIST_SKIP_CUDA_TOOLKIT)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  # ... rest of CUDA-specific build
endif()
```

4. **lib/polygeist/ExecutionEngine/CMakeLists.txt:**
```cmake
if(POLYGEIST_ENABLE_CUDA AND NOT POLYGEIST_SKIP_CUDA_TOOLKIT)
  find_package(CUDA REQUIRED)
  # ... build execution engine wrappers
endif()
```

**Build with:**
```bash
cmake .. -DPOLYGEIST_ENABLE_CUDA_SYNTAX_ONLY=ON
```

**Pros:**
- No CUDA toolkit required
- Clean, maintainable solution
- Could be upstreamed to Polygeist project
- Documents our specific use case

**Cons:**
- Requires modifying Polygeist (4 files)
- Need to maintain modifications across Polygeist updates
- More complex initial setup

**Time:** ~2-3 hours (implementation + testing)

---

### Option 3: Use Preprocessor Headers Only (Workaround)

Skip Polygeist entirely for CUDA syntax - use custom headers that replace `__global__` etc.

**Implementation:**

Create `runtime/include/hip/hip_device.h`:
```cpp
#define __global__ __attribute__((annotate("kernel")))
#define __device__ __attribute__((annotate("device")))
#define threadIdx __builtin_hip_threadIdx
// etc...
```

Modify HIP source before Polygeist:
```bash
# Preprocess to expand kernel syntax
cpp -I runtime/include kernel.hip > kernel_expanded.cpp

# Compile with Polygeist (no CUDA support needed)
cgeist kernel_expanded.cpp --mlir-output
```

**Pros:**
- No Polygeist modifications
- No CUDA toolkit needed

**Cons:**
- Kernel launch syntax `<<<>>>` cannot be handled via preprocessor
- Would need custom parsing/transformation
- Not a complete solution
- Fragile and hacky

**Status:** Not recommended

---

### Option 4: Use `.cu` Extension with Mock CUDA Headers (Temporary)

Create minimal mock CUDA headers that satisfy CMake find_package but don't actually require CUDA.

**Not Recommended:** Complex, fragile, and still requires modifying Polygeist's CMake.

---

## Recommendation

**For Development:** Option 1 (Install CUDA toolkit)
- Fastest to get working
- Standard configuration
- Can be done immediately

**For Production:** Option 2 (New CMake option)
- Cleaner long-term solution
- Documents our use case
- Could submit upstream to Polygeist

**Suggested Approach:**
1. Week 1, Tuesday: Install CUDA toolkit, rebuild Polygeist (Option 1)
2. Verify HIP syntax parsing works
3. Week 1, Wednesday-Friday: Implement Option 2 as improvement
4. Submit CMake changes upstream to Polygeist project

---

## Testing After Fix

Once Polygeist is rebuilt with CUDA support:

```bash
# Test 1: Simple CUDA kernel
cgeist hip_tests/simple_kernel.hip --cuda-lower -S

# Should produce MLIR with gpu.thread_id, no errors

# Test 2: Kernel with launch
cgeist hip_tests/simple_with_launch.cu --cuda-lower -S

# Should produce MLIR with gpu.launch_func

# Test 3: With our hip_runtime.h header
cgeist hip_tests/simple_malloc_test.hip \
    -I runtime/include \
    --cuda-lower -S

# Should show both vx_mem_alloc calls AND gpu.launch_func
```

---

## Decision Required

**Question:** Which option should we proceed with?

**Factors to consider:**
- Time available (Week 1, Tuesday)
- Long-term maintenance
- Whether to contribute back to Polygeist
- System constraints (disk space, permissions)

---

**Status:** Awaiting decision on approach
**Assigned To:** Developer Team
**Priority:** HIGH - blocks Week 1 Tuesday testing
