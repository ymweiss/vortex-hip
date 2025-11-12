# Metadata Generation Testing

This directory contains tests for the HIP kernel metadata generation system (Phase 1 prototype).

## Files

- **`test_script.sh`** - Automated test suite for metadata generator
- **`test_manual_metadata.cpp`** - Manual metadata test demonstrating registration
- **`simple_kernel.cpp`** - Sample kernel source for testing (requires RISC-V compiler)

## Running Tests

### Quick Test

```bash
./test_script.sh
```

This validates:
- Script installation and execution
- Error handling
- Output format
- Compilation of generated metadata

### Manual Metadata Test

Test the registration and marshaling system with pre-defined metadata:

```bash
g++ test_manual_metadata.cpp \
    -I$HOME/vortex_hip/runtime/include \
    -L$HOME/vortex_hip/runtime/build \
    -lhip_vortex \
    -Wl,-rpath,$HOME/vortex_hip/runtime/build \
    -o test_manual_metadata

./test_manual_metadata
```

Expected output:
```
=== Manual Metadata Test ===
Expected metadata for vectorAdd(float* a, float* b, float* c, int n):
  arg[0]: offset=0,  size=8, align=8, pointer=1 (float* a)
  arg[1]: offset=8,  size=8, align=8, pointer=1 (float* b)
  arg[2]: offset=16, size=8, align=8, pointer=1 (float* c)
  arg[3]: offset=24, size=4, align=4, pointer=0 (int n)
```

Note: "hipErrorNotInitialized" is expected - this test validates metadata format without requiring a Vortex device.

## Full Workflow Test (Requires RISC-V Compiler)

If you have a RISC-V compiler installed:

```bash
# 1. Compile kernel to ELF with debug info
riscv64-unknown-elf-gcc -g simple_kernel.cpp -o simple_kernel.elf

# 2. Generate metadata
python3 $HOME/vortex/scripts/hip_metadata_gen.py simple_kernel.elf > kernel_stub.cpp

# 3. Inspect generated metadata
cat kernel_stub.cpp

# 4. Compile stub
g++ -c kernel_stub.cpp -I$HOME/vortex_hip/runtime/include

# 5. Link with application (requires kernel binary from vxbin.py)
```

## Test Results

All tests passing indicate:
- ✓ Python script is properly installed
- ✓ Metadata structure is correct
- ✓ Registration function works
- ✓ Generated code compiles successfully
- ✓ Integration with runtime is functional

## Next Steps

After validating Phase 1:
1. Test with real Vortex-compiled kernels
2. Integrate into Vortex build system
3. Proceed with Phase 2 (LLVM pass) implementation

## Related Documentation

- `$HOME/vortex/scripts/README_metadata_gen.md` - Script documentation
- `$HOME/vortex_hip/runtime/ARGUMENT_MARSHALING.md` - Runtime implementation
- `$HOME/vortex_hip/docs/implementation/COMPILER_METADATA_GENERATION.md` - Full strategy
