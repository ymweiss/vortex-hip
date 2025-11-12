# Phase 2: Compiler Integration

**Status:** 📋 PLANNED (LLVM submodule available)

## Overview

Phase 2 will integrate metadata generation directly into the LLVM compiler, eliminating the need for separate DWARF parsing. Metadata will be generated automatically during kernel compilation.

## Approach

### Option 1: Clang Plugin (Recommended)
- **Complexity:** Medium
- **Integration:** External plugin loaded by Clang
- **Pros:** No LLVM modifications, easier to maintain
- **Cons:** Requires plugin infrastructure

### Option 2: LLVM Pass
- **Complexity:** Medium-High
- **Integration:** Custom LLVM optimization pass
- **Pros:** Standard LLVM workflow
- **Cons:** Requires LLVM build modifications

### Option 3: CodeGen Backend
- **Complexity:** High
- **Integration:** Deep into LLVM backend
- **Pros:** Most control, optimal code generation
- **Cons:** Complex, harder to maintain

## Components

### LLVM Submodule
- **Location:** `llvm-vortex/` (git submodule)
- **Repository:** https://github.com/vortexgpgpu/llvm
- **Version:** LLVM 18.1.7 (custom fork with Vortex extensions)
- **Size:** 940MB

### Implementation Tasks

1. **Create Metadata Pass/Plugin**
   - Detect HIP kernel functions (`__global__`)
   - Extract function signature
   - Analyze parameter types
   - Generate metadata structure

2. **Emit Metadata**
   - Generate C++ registration code
   - Or emit metadata in ELF section
   - Link with kernel binary

3. **Integration**
   - Update Vortex toolchain scripts
   - Modify hipcc wrapper
   - Update build system

## Reference Implementation

Phase 1's Python script serves as reference for what the compiler needs to generate:

```cpp
// What the compiler should emit:
static const hipKernelArgumentMetadata kernel_metadata[] = {
    {.offset = 0,  .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 4,  .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 8,  .size = 4, .alignment = 4, .is_pointer = 1},
    {.offset = 12, .size = 4, .alignment = 4, .is_pointer = 0}
};

__attribute__((constructor))
static void register_kernel() {
    __hipRegisterFunctionWithMetadata(&handle, "kernel", binary, size, 4, kernel_metadata);
}
```

## Files

### Submodule
- `llvm-vortex/` - LLVM source (940MB)

### Documentation
- `docs/implementation/COMPILER_INFRASTRUCTURE.md` - Detailed design
- `docs/implementation/IMPLEMENTATION_CHECKLIST.md` - Step-by-step guide
- `docs/implementation/COMPILER_METADATA_GENERATION.md` - Technical specs

## Development Status

- ✅ LLVM submodule integrated
- ✅ Reference implementation (Python) complete
- ✅ Metadata format defined and tested
- ⏳ Compiler pass implementation - Not started
- ⏳ Integration with hipcc - Not started
- ⏳ Testing infrastructure - Not started

## Prerequisites

- LLVM development experience
- Understanding of Clang AST/IR
- Familiarity with HIP compilation model
- Phase 1 complete (reference implementation)

## Timeline

Estimated: 6-8 weeks
- Week 1-2: Study LLVM infrastructure, design pass
- Week 3-4: Implement metadata generation
- Week 5-6: Integration and testing
- Week 7-8: Optimization and documentation

## See Also

- Phase 1 (python script) for reference implementation
- `docs/implementation/COMPILER_INFRASTRUCTURE.md` for technical details
- Vortex LLVM documentation for target-specific extensions
