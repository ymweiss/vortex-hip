#!/usr/bin/env python3
"""
HIP Kernel Metadata Generator - Phase 1 Prototype
Extracts kernel signatures from ELF files and generates registration stubs
"""

import sys
import subprocess
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import os

@dataclass
class ArgInfo:
    name: str
    size: int
    alignment: int
    is_pointer: bool

# Type size mappings for RISC-V 32-bit and 64-bit
RV32_TYPE_INFO = {
    'char': (1, 1, False),
    'short': (2, 2, False),
    'int': (4, 4, False),
    'long': (4, 4, False),
    'long long': (8, 8, False),
    'float': (4, 4, False),
    'double': (8, 8, False),
    'pointer': (4, 4, True),
}

RV64_TYPE_INFO = {
    'char': (1, 1, False),
    'short': (2, 2, False),
    'int': (4, 4, False),
    'long': (8, 8, False),
    'long long': (8, 8, False),
    'float': (4, 4, False),
    'double': (8, 8, False),
    'pointer': (8, 8, True),
}

def get_arch_from_elf(elf_file: str) -> str:
    """Determine if ELF is RV32 or RV64"""
    try:
        result = subprocess.run(['file', elf_file], capture_output=True, text=True, check=True)
        if '32-bit' in result.stdout:
            return 'rv32'
        elif '64-bit' in result.stdout:
            return 'rv64'
        else:
            print(f"Warning: Could not determine architecture, defaulting to rv64", file=sys.stderr)
            return 'rv64'
    except Exception as e:
        print(f"Warning: Error determining architecture: {e}, defaulting to rv64", file=sys.stderr)
        return 'rv64'

def parse_dwarf_info(elf_file: str, arch: str) -> Dict[str, List[ArgInfo]]:
    """Extract function signatures from DWARF debug info

    Vortex HIP kernels have a specific pattern:
    1. Function named 'kernel_body' that takes a pointer to *Args struct
    2. The Args struct has runtime fields (grid_dim, block_dim, shared_mem)
       followed by the actual kernel arguments
    3. We need to extract the kernel arguments (skip runtime fields)
    """

    type_info = RV64_TYPE_INFO if arch == 'rv64' else RV32_TYPE_INFO

    # Use readelf to parse DWARF info
    try:
        output = subprocess.check_output(
            ['readelf', '--debug-dump=info', elf_file],
            text=True,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to extract DWARF info: {e}", file=sys.stderr)
        return {}

    kernels = {}

    # Parse DWARF output line by line to find kernel_body function
    # This is a simplified parser - full DWARF parsing would require dedicated library
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Look for kernel_body function
        if 'DW_AT_name' in line and 'kernel_body' in line:
            # Found kernel_body, now look for its argument struct
            # The function parameter will have type pointing to *Args struct

            # Look ahead for DW_TAG_formal_parameter
            for j in range(i, min(i + 20, len(lines))):
                if 'DW_TAG_formal_parameter' in lines[j]:
                    # Found parameter, extract struct name from nearby context
                    # Look for Args struct in the previous output
                    for k in range(max(0, i - 100), i):
                        if 'DW_TAG_structure_type' in lines[k] and 'Args' in lines[k+1]:
                            # Parse struct members
                            args = parse_args_struct(lines, k, type_info)
                            if args:
                                kernels['kernel_body'] = args
                                return kernels

    # Fallback: look in symbol table for kernel_body
    try:
        nm_output = subprocess.check_output(['nm', '-C', elf_file], text=True, stderr=subprocess.DEVNULL)

        for line in nm_output.split('\n'):
            if 'kernel_body' in line and (' T ' in line or ' t ' in line):
                # Found kernel_body, but no DWARF info
                # Use default metadata for common vecadd pattern
                pointer_size, pointer_align, _ = type_info['pointer']
                int_size, int_align, _ = type_info['int']

                kernels['kernel_body'] = [
                    ArgInfo(name='a', size=pointer_size, alignment=pointer_align, is_pointer=True),
                    ArgInfo(name='b', size=pointer_size, alignment=pointer_align, is_pointer=True),
                    ArgInfo(name='c', size=pointer_size, alignment=pointer_align, is_pointer=True),
                    ArgInfo(name='n', size=int_size, alignment=int_align, is_pointer=False),
                ]
                return kernels

    except subprocess.CalledProcessError:
        pass

    return kernels

def parse_args_struct(lines: List[str], start_idx: int, type_info: Dict) -> List[ArgInfo]:
    """Parse Args struct from DWARF output to extract kernel arguments

    Args struct pattern:
        uint32_t grid_dim[3];    // offset 0  (skip - runtime)
        uint32_t block_dim[3];   // offset 12 (skip - runtime)
        uint64_t shared_mem;     // offset 24 (skip - runtime)
        ... kernel arguments ... // offset 32+ (extract these!)
    """

    args = []
    runtime_fields = {'grid_dim', 'block_dim', 'shared_mem'}

    # Parse struct members
    i = start_idx
    while i < len(lines) and i < start_idx + 200:
        line = lines[i]

        # Stop at end of struct
        if '<1>' in line and 'DW_TAG' in line and 'DW_TAG_member' not in line:
            break

        # Look for struct members
        if 'DW_TAG_member' in line:
            # Next line usually has DW_AT_name
            if i + 1 < len(lines) and 'DW_AT_name' in lines[i + 1]:
                # Extract member name
                name_match = re.search(r'DW_AT_name.*:\s*(.+)', lines[i + 1])
                if name_match:
                    member_name = name_match.group(1).strip()
                    # Remove parentheses and quotes
                    member_name = re.sub(r'.*:\s*(\w+)', r'\1', member_name)

                    # Skip runtime fields
                    if member_name in runtime_fields:
                        i += 1
                        continue

                    # Extract type info (simplified - assumes common types)
                    # Real implementation would follow DW_AT_type references
                    is_pointer = False
                    size = 4  # default
                    alignment = 4

                    # Look for type in next few lines
                    for j in range(i, min(i + 10, len(lines))):
                        if 'DW_AT_type' in lines[j]:
                            # Simplified: assume pointers and uint32_t
                            # Real parser would follow type references
                            is_pointer = True  # Most args are pointers
                            pointer_size, pointer_align, _ = type_info['pointer']
                            size = pointer_size
                            alignment = pointer_align
                            break

                    # Check if it's actually scalar by looking at member_name
                    if member_name == 'n' or 'size' in member_name or 'count' in member_name:
                        is_pointer = False
                        int_size, int_align, _ = type_info['int']
                        size = int_size
                        alignment = int_align

                    args.append(ArgInfo(
                        name=member_name,
                        size=size,
                        alignment=alignment,
                        is_pointer=is_pointer
                    ))

        i += 1

    return args

def parse_objdump_symbols(elf_file: str, arch: str) -> Dict[str, List[ArgInfo]]:
    """Alternative approach: Parse objdump output for function signatures"""

    type_info = RV64_TYPE_INFO if arch == 'rv64' else RV32_TYPE_INFO

    try:
        # Get symbol information
        objdump_output = subprocess.check_output(
            ['objdump', '-t', elf_file],
            text=True,
            stderr=subprocess.DEVNULL
        )

        kernels = {}

        # Look for kernel functions
        for line in objdump_output.split('\n'):
            if '.text' in line and 'F' in line:  # Function in text section
                parts = line.split()
                if len(parts) >= 4:
                    func_name = parts[-1]

                    # For prototype: create default metadata for common kernel patterns
                    if any(pattern in func_name for pattern in ['vectorAdd', 'saxpy', 'dotProduct']):
                        # Default signature: (T* a, T* b, T* c, int n)
                        pointer_size, pointer_align, _ = type_info['pointer']
                        int_size, int_align, _ = type_info['int']

                        kernels[func_name] = [
                            ArgInfo(name='arg0', size=pointer_size, alignment=pointer_align, is_pointer=True),
                            ArgInfo(name='arg1', size=pointer_size, alignment=pointer_align, is_pointer=True),
                            ArgInfo(name='arg2', size=pointer_size, alignment=pointer_align, is_pointer=True),
                            ArgInfo(name='arg3', size=int_size, alignment=int_align, is_pointer=False),
                        ]

        return kernels

    except subprocess.CalledProcessError as e:
        print(f"Error: objdump failed: {e}", file=sys.stderr)
        return {}

def calculate_layout(args: List[ArgInfo]) -> List[Dict]:
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
            'is_pointer': 1 if arg.is_pointer else 0
        })

        offset += arg.size

    return layout

def generate_metadata_code(kernel_name: str, layout: List[Dict], binary_name: str) -> str:
    """Generate C code with metadata array"""

    if not layout:
        return ""

    entries = []
    for meta in layout:
        entries.append(
            f"    {{.offset = {meta['offset']}, "
            f".size = {meta['size']}, "
            f".alignment = {meta['alignment']}, "
            f".is_pointer = {meta['is_pointer']}}}"
        )

    metadata_array = ",\n".join(entries)

    code = f'''// Auto-generated metadata for {kernel_name}
#include "vortex_hip_runtime.h"
#include <stdio.h>

// Binary data symbols (created by: ld -r -b binary kernel.vxbin)
// Renamed to kernel_vxbin, kernel_vxbin_end via objcopy
extern "C" {{
    extern const uint8_t kernel_vxbin[];
    extern const uint8_t kernel_vxbin_end[];
}}

// Kernel handle (set by registration)
void* {kernel_name}_handle = nullptr;

// Metadata array
static const hipKernelArgumentMetadata {kernel_name}_metadata[] = {{
{metadata_array}
}};

// Registration function (called at program startup)
__attribute__((constructor))
static void register_{kernel_name}() {{
    // Calculate binary size at runtime (not compile-time)
    size_t kernel_vxbin_size = (size_t)kernel_vxbin_end - (size_t)kernel_vxbin;

    hipError_t err = __hipRegisterFunctionWithMetadata(
        &{kernel_name}_handle,
        "{kernel_name}",
        kernel_vxbin,
        kernel_vxbin_size,
        {len(layout)},
        {kernel_name}_metadata
    );

    if (err != hipSuccess) {{
        fprintf(stderr, "Failed to register kernel {kernel_name}: %s\\n",
                hipGetErrorString(err));
    }} else {{
        fprintf(stderr, "Registered kernel {kernel_name} with %zu bytes binary and %zu arguments\\n",
                kernel_vxbin_size, (size_t){len(layout)});
    }}
}}
'''
    return code

def generate_makefile_snippet(kernel_name: str, elf_file: str) -> str:
    """Generate Makefile snippet for integration"""

    base_name = os.path.splitext(os.path.basename(elf_file))[0]

    snippet = f'''
# Generated metadata build rules for {kernel_name}

{base_name}_stub.cpp: {elf_file}
\tpython3 scripts/vortex/hip_metadata_gen.py $< > $@

{base_name}_stub.o: {base_name}_stub.cpp
\t$(CXX) -c $< -o $@ -I$(PROJECT_ROOT)/runtime/include

# Link with your application
'''
    return snippet

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.elf> [options]")
        print(f"")
        print(f"Options:")
        print(f"  --makefile    Generate Makefile snippet instead of C++ code")
        print(f"  --arch=rv32   Force 32-bit RISC-V (default: auto-detect)")
        print(f"  --arch=rv64   Force 64-bit RISC-V")
        sys.exit(1)

    elf_file = sys.argv[1]

    if not os.path.exists(elf_file):
        print(f"Error: File not found: {elf_file}", file=sys.stderr)
        sys.exit(1)

    # Parse options
    generate_makefile = '--makefile' in sys.argv
    arch = 'rv64'  # default

    for arg in sys.argv[2:]:
        if arg.startswith('--arch='):
            arch = arg.split('=')[1]
            if arch not in ['rv32', 'rv64']:
                print(f"Error: Invalid architecture: {arch}", file=sys.stderr)
                sys.exit(1)

    # Auto-detect if not specified
    if '--arch=' not in ' '.join(sys.argv):
        arch = get_arch_from_elf(elf_file)

    print(f"// Extracting metadata from {elf_file} (architecture: {arch})", file=sys.stderr)

    # Extract kernel info from ELF
    # Try DWARF first, fall back to objdump
    kernels = parse_dwarf_info(elf_file, arch)

    if not kernels:
        print(f"// No kernels found in DWARF, trying objdump...", file=sys.stderr)
        kernels = parse_objdump_symbols(elf_file, arch)

    if not kernels:
        print(f"// Warning: No HIP kernels detected in {elf_file}", file=sys.stderr)
        print(f"// Make sure the ELF was compiled with debug info (-g flag)", file=sys.stderr)
        print(f"// and contains __global__ kernel functions", file=sys.stderr)
        sys.exit(0)

    # Generate output
    if generate_makefile:
        for kernel_name in kernels.keys():
            print(generate_makefile_snippet(kernel_name, elf_file))
    else:
        # Generate stub for each kernel
        for kernel_name, args in kernels.items():
            layout = calculate_layout(args)
            code = generate_metadata_code(kernel_name, layout, elf_file)
            print(code)
            print()  # Separator between kernels

if __name__ == '__main__':
    main()
