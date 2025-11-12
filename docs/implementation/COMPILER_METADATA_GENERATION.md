# Compiler Metadata Generation for HIP Kernels

## Overview

This document describes the strategy for modifying the Vortex LLVM compiler to automatically generate argument metadata for HIP kernels, enabling correct argument marshaling at runtime.

## Current Compilation Flow

### Vortex Kernel Compilation (Native)

```
kernel.cpp
    ↓
LLVM Vortex Clang
    ↓ (RISC-V target)
kernel.elf
    ↓ (vxbin.py script)
kernel.vxbin (Vortex binary format)
```

### HIP Kernel Compilation (Desired)

```
kernel.hip
    ↓
HIP-aware Clang
    ↓ (Extract kernel signatures)
    ├─→ kernel.vxbin (device code)
    └─→ stub.cpp (host registration code WITH METADATA)
         ↓
    combined.o
```

## The Problem

**Current State:**
- Vortex compiles kernels to RISC-V binaries
- No automatic HIP kernel detection
- No metadata generation
- Manual struct packing required

**What We Need:**
- Detect `__global__` kernel functions
- Extract argument types, sizes, alignments
- Generate `hipKernelArgumentMetadata` arrays
- Generate registration code calling `__hipRegisterFunctionWithMetadata()`

## Solution: Multi-Stage Approach

### Stage 1: Python Script Post-Processing (Immediate)

**Pros:** Quick to implement, no compiler changes
**Cons:** Limited type information, fragile parsing

```python
# hip_metadata_gen.py - Extract from DWARF debug info

import subprocess
import re

def extract_kernel_metadata(elf_file):
    # 1. Use objdump to find __global__ functions
    # 2. Parse DWARF info for parameter types
    # 3. Calculate sizes and alignments
    # 4. Generate C code with metadata arrays
    pass

def generate_registration_stub(kernel_name, metadata):
    return f'''
    static const hipKernelArgumentMetadata {kernel_name}_metadata[] = {{
        {generate_metadata_entries(metadata)}
    }};

    __attribute__((constructor))
    static void register_{kernel_name}() {{
        __hipRegisterFunctionWithMetadata(
            &{kernel_name}_handle,
            "{kernel_name}",
            {kernel_name}_vxbin,
            {kernel_name}_vxbin_size,
            {len(metadata)},
            {kernel_name}_metadata
        );
    }}
    '''
```

**Usage:**
```bash
# Compile kernel
clang++ -target riscv32 kernel.cpp -o kernel.elf

# Extract metadata
python3 hip_metadata_gen.py kernel.elf > kernel_stub.cpp

# Compile host code with stub
clang++ main.cpp kernel_stub.cpp -o app
```

### Stage 2: LLVM Pass (Better)

**Pros:** Full type information, compiler integration
**Cons:** Requires LLVM pass development

```cpp
// HipMetadataGenPass.cpp - LLVM optimization pass

class HipMetadataGenPass : public ModulePass {
public:
    bool runOnModule(Module &M) override {
        for (Function &F : M) {
            if (isHipKernel(F)) {
                auto metadata = extractArgumentMetadata(F);
                generateRegistrationCode(F, metadata);
            }
        }
        return true;
    }

private:
    bool isHipKernel(Function &F) {
        // Check for __global__ attribute
        return F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
               F.hasFnAttribute("hip-kernel");
    }

    std::vector<ArgMetadata> extractArgumentMetadata(Function &F) {
        std::vector<ArgMetadata> metadata;
        const DataLayout &DL = F.getParent()->getDataLayout();

        for (Argument &Arg : F.args()) {
            ArgMetadata meta;
            Type *Ty = Arg.getType();

            meta.size = DL.getTypeAllocSize(Ty);
            meta.alignment = DL.getABITypeAlignment(Ty);
            meta.is_pointer = Ty->isPointerTy();

            metadata.push_back(meta);
        }

        return metadata;
    }
};
```

**Integration:**
```bash
# Add pass to compilation
clang++ -Xclang -load -Xclang libHipMetadataGen.so \
        -mllvm -hip-gen-metadata \
        kernel.cpp -o kernel.o
```

### Stage 3: Clang Plugin (Best)

**Pros:** Full integration, clean interface, compiler-aware
**Cons:** Most development effort

```cpp
// HipMetadataPlugin.cpp - Clang plugin

class HipMetadataConsumer : public ASTConsumer {
public:
    void HandleTranslationUnit(ASTContext &Context) override {
        // Walk AST looking for __global__ functions
        TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (isHipKernel(FD)) {
            generateMetadata(FD);
            generateStub(FD);
        }
        return true;
    }

    bool isHipKernel(FunctionDecl *FD) {
        // Check for __global__ attribute
        return FD->hasAttr<CUDAGlobalAttr>();
    }

    void generateMetadata(FunctionDecl *FD) {
        for (ParmVarDecl *Param : FD->parameters()) {
            QualType Type = Param->getType();

            // Get type info from ASTContext
            int64_t size = Context.getTypeSize(Type) / 8;
            int64_t align = Context.getTypeAlign(Type) / 8;
            bool is_pointer = Type->isPointerType();

            // Emit metadata entry
            emitMetadataEntry(Param->getName(), size, align, is_pointer);
        }
    }
};
```

## Recommended Implementation: Hybrid Approach

### Phase 1: Python Prototype (Week 1)

Build a working prototype using Python + DWARF:

**File:** `hip_metadata_gen.py`

```python
#!/usr/bin/env python3
"""
HIP Kernel Metadata Generator
Extracts kernel signatures from ELF files and generates registration stubs
"""

import sys
import subprocess
import re
from dataclasses import dataclass
from typing import List

@dataclass
class ArgInfo:
    name: str
    size: int
    alignment: int
    is_pointer: bool

def parse_dwarf_info(elf_file: str) -> dict:
    """Extract function signatures from DWARF debug info"""
    # Use llvm-dwarfdump or readelf
    cmd = ['llvm-dwarfdump', '--debug-info', elf_file]
    output = subprocess.check_output(cmd, text=True)

    kernels = {}
    # Parse DWARF DW_TAG_subprogram with __global__ attribute
    # Extract DW_TAG_formal_parameter entries
    # Calculate sizes from DW_AT_type

    return kernels

def calculate_layout(args: List[ArgInfo]) -> List[dict]:
    """Calculate argument buffer layout with proper alignment"""
    offset = 0
    layout = []

    for arg in args:
        # Align offset
        padding = (arg.alignment - (offset % arg.alignment)) % arg.alignment
        offset += padding

        layout.append({
            'name': arg.name,
            'offset': offset,
            'size': arg.size,
            'alignment': arg.alignment,
            'is_pointer': int(arg.is_pointer)
        })

        offset += arg.size

    return layout

def generate_metadata_code(kernel_name: str, layout: List[dict]) -> str:
    """Generate C code with metadata array"""
    entries = []
    for meta in layout:
        entries.append(
            f"    {{.offset = {meta['offset']}, "
            f".size = {meta['size']}, "
            f".alignment = {meta['alignment']}, "
            f".is_pointer = {meta['is_pointer']}}}"
        )

    metadata_array = ",\n".join(entries)

    code = f'''
// Auto-generated metadata for {kernel_name}
#include "vortex_hip_runtime.h"

extern const uint8_t {kernel_name}_vxbin[];
extern const size_t {kernel_name}_vxbin_size;

static void* {kernel_name}_handle = nullptr;

static const hipKernelArgumentMetadata {kernel_name}_metadata[] = {{
{metadata_array}
}};

__attribute__((constructor))
static void register_{kernel_name}() {{
    __hipRegisterFunctionWithMetadata(
        &{kernel_name}_handle,
        "{kernel_name}",
        {kernel_name}_vxbin,
        {kernel_name}_vxbin_size,
        {len(layout)},
        {kernel_name}_metadata
    );
}}
'''
    return code

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <kernel.elf>")
        sys.exit(1)

    elf_file = sys.argv[1]

    # Extract kernel info from DWARF
    kernels = parse_dwarf_info(elf_file)

    # Generate stub for each kernel
    for kernel_name, args in kernels.items():
        layout = calculate_layout(args)
        code = generate_metadata_code(kernel_name, layout)
        print(code)

if __name__ == '__main__':
    main()
```

**Build Integration:**
```makefile
# Makefile snippet
%.stub.cpp: %.elf
	python3 $(VORTEX_HOME)/scripts/hip_metadata_gen.py $< > $@

%.o: %.stub.cpp
	$(CXX) -c $< -o $@
```

### Phase 2: LLVM Pass Integration (Week 2-3)

Develop proper LLVM pass for metadata extraction.

**Location:** `llvm-vortex/lib/Target/RISCV/HipMetadataGen.cpp`

**Key APIs:**
- `DataLayout::getTypeAllocSize()` - Get type size
- `DataLayout::getABITypeAlignment()` - Get alignment
- `Type::isPointerTy()` - Check if pointer
- `Function::arg_begin()` - Iterate arguments

### Phase 3: Full Clang Integration (Week 4+)

Build Clang plugin for seamless HIP support.

## Type Size Reference (RV32)

```
Type               | Size | Alignment | is_pointer
-------------------+------+-----------+-----------
char               | 1    | 1         | 0
short              | 2    | 2         | 0
int                | 4    | 4         | 0
long               | 4    | 4         | 0
long long          | 8    | 8         | 0
float              | 4    | 4         | 0
double             | 8    | 8         | 0
pointer (any*)     | 4    | 4         | 1
struct (packed)    | sum  | 1         | 0
struct (normal)    | sum  | max field | 0
```

## Type Size Reference (RV64)

```
Type               | Size | Alignment | is_pointer
-------------------+------+-----------+-----------
char               | 1    | 1         | 0
short              | 2    | 2         | 0
int                | 4    | 4         | 0
long               | 8    | 8         | 0
long long          | 8    | 8         | 0
float              | 4    | 4         | 0
double             | 8    | 8         | 0
pointer (any*)     | 8    | 8         | 1
struct (packed)    | sum  | 1         | 0
struct (normal)    | sum  | max field | 0
```

## Testing Strategy

### Test 1: Simple Kernel
```cpp
__global__ void simple(int* data, int n) { }
// Expected: [{offset:0, size:8, align:8, ptr:1},
//            {offset:8, size:4, align:4, ptr:0}]
```

### Test 2: Mixed Types
```cpp
__global__ void mixed(float* a, int b, double* c, char d) { }
// Expected: proper alignment and padding
```

### Test 3: Struct by Value
```cpp
struct Vec3 { float x, y, z; };
__global__ void structured(Vec3 v, int n) { }
// Expected: 12-byte struct, 4-byte int
```

## Files to Create/Modify

### New Files
1. **`vortex/scripts/hip_metadata_gen.py`** - Python extractor
2. **`vortex_hip/compiler/README.md`** - Compiler integration guide
3. **`llvm-vortex/lib/Target/RISCV/HipMetadataGen.cpp`** - LLVM pass (future)

### Modified Files
1. **`vortex/tests/regression/common.mk`** - Add metadata generation step
2. **`vortex/kernel/scripts/vxbin.py`** - Optionally embed metadata in binary

## Next Steps

1. **Create Python prototype** - Working reference implementation
2. **Test with existing HIP kernels** - Validate metadata accuracy
3. **Integrate with build system** - Automatic generation
4. **Develop LLVM pass** - Proper compiler integration
5. **Add to hipcc wrapper** - Seamless user experience

## References

- **DWARF Standard:** Debug info format
- **LLVM DataLayout:** Type size/alignment APIs
- **Clang AST:** Syntax tree traversal
- **Vortex ABI:** RISC-V calling convention

---

**Status:** Design Complete
**Next Action:** Implement Python prototype
**Timeline:** 1-2 weeks for working prototype
