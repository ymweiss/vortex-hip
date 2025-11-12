# Quick Start: Building Vortex HIP Runtime

## Prerequisites

1. Vortex GPU installed at `~/vortex` (or set `VORTEX_ROOT`)
2. CMake 3.10+
3. C++14 compiler (GCC 7+ or Clang 5+)

## Build in 3 Steps

```bash
# 1. Navigate to runtime directory
cd ~/vortex_hip/runtime

# 2. Configure and build
mkdir build && cd build
cmake .. -DVORTEX_ROOT=~/vortex
make -j$(nproc)

# 3. Set library path and run example
export LD_LIBRARY_PATH=~/vortex/build/runtime:$LD_LIBRARY_PATH
./examples/vector_add
```

**Note**: If Vortex is not built yet:
```bash
cd ~/vortex/build
make -j$(nproc)
```

## Expected Output

```
Vortex HIP Vector Addition Example
===================================

Number of devices: 1

Device 0: Vortex RISC-V GPU
  Compute capability: 1.0
  Total global memory: 256 MB
  ...

SUCCESS: All results match!
```

## What You Get

✅ **libvortex_hip.so** - HIP runtime library
✅ **Headers** - `vortex_hip_runtime.h` and `vortex_hip_device.h`
✅ **Examples** - Vector addition and more

## Using in Your Project

### Option 1: CMake

```cmake
find_package(vortex_hip REQUIRED)
target_link_libraries(your_app vortex_hip::vortex_hip)
```

### Option 2: Manual

```bash
g++ your_app.cpp \
    -I~/vortex_hip/runtime/include \
    -L~/vortex_hip/runtime/build \
    -lvortex_hip -lvortex \
    -o your_app
```

## Next Steps

- Read [README.md](README.md) for full documentation
- See [HIP-TO-VORTEX-API-MAPPING.md](../docs/implementation/HIP-TO-VORTEX-API-MAPPING.md) for API details
- Check [examples/](examples/) for more code samples

## Troubleshooting

**Problem**: `vortex.h not found`
```bash
export VORTEX_ROOT=~/vortex
cmake .. -DVORTEX_ROOT=$VORTEX_ROOT
```

**Problem**: `libvortex.so not found`
```bash
export LD_LIBRARY_PATH=~/vortex/stub:$LD_LIBRARY_PATH
```

**Problem**: Build fails
```bash
# Make sure Vortex is built first
cd ~/vortex/build
make -j$(nproc)
```
