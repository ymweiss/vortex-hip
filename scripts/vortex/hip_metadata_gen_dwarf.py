#!/usr/bin/env python3
"""
HIP Kernel Metadata Generator - Phase 1 with DWARF Parsing
Extracts kernel signatures from ELF DWARF debug info
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
def get_pointer_size(arch: str) -> int:
    return 8 if arch == 'rv64' else 4

def get_arch_from_elf(elf_file: str) -> str:
    """Determine if ELF is RV32 or RV64"""
    try:
        result = subprocess.run(['file', elf_file], capture_output=True, text=True, check=True)
        if '32-bit' in result.stdout:
            return 'rv32'
        elif '64-bit' in result.stdout:
            return 'rv64'
        else:
            return 'rv64'
    except Exception:
        return 'rv64'

def parse_dwarf_struct(dwarf_output: str, struct_name: str, arch: str) -> Optional[List[ArgInfo]]:
    """Parse DWARF output to extract struct member information"""

    args = []
    lines = dwarf_output.split('\n')

    # Find the struct definition
    in_struct = False
    struct_pattern = f'DW_TAG_structure_type.*DW_AT_name.*"{struct_name}"'
    member_pattern = r'DW_TAG_member'
    name_pattern = r'DW_AT_name\s+\("([^"]+)"\)'
    type_ref_pattern = r'DW_AT_type\s+\(0x([0-9a-f]+)'
    offset_pattern = r'DW_AT_data_member_location\s+\(0x([0-9a-f]+)\)'

    # First pass: find struct and collect member info
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if we found the struct
        if re.search(struct_pattern, line):
            in_struct = True
            i += 1
            continue

        # If in struct, look for members
        if in_struct and 'DW_TAG_member' in line:
            member_name = None
            member_type_ref = None
            member_offset = None

            # Parse member attributes
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('0x') and 'NULL' not in lines[j]:
                name_match = re.search(name_pattern, lines[j])
                if name_match:
                    member_name = name_match.group(1)

                type_match = re.search(type_ref_pattern, lines[j])
                if type_match:
                    member_type_ref = type_match.group(1)

                offset_match = re.search(offset_pattern, lines[j])
                if offset_match:
                    member_offset = int(offset_match.group(1), 16)

                j += 1

            # Skip runtime-added fields (grid_dim, block_dim, shared_mem)
            if member_name and member_name not in ['grid_dim', 'block_dim', 'shared_mem']:
                if member_type_ref:
                    # Look up the type to get size and check if pointer
                    type_info = find_type_info(dwarf_output, member_type_ref)
                    if type_info:
                        args.append(ArgInfo(
                            name=member_name,
                            size=type_info['size'],
                            alignment=type_info['alignment'],
                            is_pointer=type_info['is_pointer']
                        ))

            i = j
            continue

        # Exit struct when we hit NULL or new top-level tag
        if in_struct and ('NULL' in line or (line.strip().startswith('0x') and 'DW_TAG' in line and 'DW_TAG_member' not in line)):
            break

        i += 1

    return args if args else None

def find_type_info(dwarf_output: str, type_ref: str) -> Optional[Dict]:
    """Find type information by reference ID"""

    lines = dwarf_output.split('\n')

    # Find the type definition
    type_addr = f'0x{type_ref.lstrip("0x").zfill(8)}:'

    for i, line in enumerate(lines):
        if line.startswith(type_addr):
            # Found the type, now extract info
            is_pointer = 'DW_TAG_pointer_type' in line
            is_typedef = 'DW_TAG_typedef' in line
            is_base = 'DW_TAG_base_type' in line

            if is_pointer:
                # Pointers: size depends on architecture
                # For RV32, pointers are 4 bytes
                # We can infer from the file architecture
                if '32' in dwarf_output[:500]:  # Check early in file for arch
                    return {'size': 4, 'alignment': 4, 'is_pointer': True}
                else:
                    return {'size': 8, 'alignment': 8, 'is_pointer': True}

            elif is_base:
                # Base type: extract size
                size_match = re.search(r'DW_AT_byte_size\s+\(0x([0-9a-f]+)\)', line)
                if size_match:
                    size = int(size_match.group(1), 16)
                    return {'size': size, 'alignment': size, 'is_pointer': False}

                # Check next few lines for byte_size
                for j in range(i+1, min(i+5, len(lines))):
                    size_match = re.search(r'DW_AT_byte_size\s+\(0x([0-9a-f]+)\)', lines[j])
                    if size_match:
                        size = int(size_match.group(1), 16)
                        return {'size': size, 'alignment': size, 'is_pointer': False}

            elif is_typedef:
                # Typedef: follow the type reference
                type_match = re.search(r'DW_AT_type\s+\(0x([0-9a-f]+)', line)
                if not type_match:
                    # Check next line
                    if i+1 < len(lines):
                        type_match = re.search(r'DW_AT_type\s+\(0x([0-9a-f]+)', lines[i+1])

                if type_match:
                    return find_type_info(dwarf_output, type_match.group(1))

            break

    # Default fallback
    return {'size': 4, 'alignment': 4, 'is_pointer': False}

def parse_dwarf_info(elf_file: str, arch: str) -> Dict[str, List[ArgInfo]]:
    """Extract kernel argument information from DWARF debug info"""

    # Use llvm-dwarfdump if available
    llvm_tools_dir = os.environ.get('LLVM_VORTEX', '')
    dwarfdump_cmd = None

    if llvm_tools_dir and os.path.exists(f"{llvm_tools_dir}/bin/llvm-dwarfdump"):
        dwarfdump_cmd = [f"{llvm_tools_dir}/bin/llvm-dwarfdump", "--debug-info", elf_file]
    elif subprocess.run(['which', 'llvm-dwarfdump'], capture_output=True).returncode == 0:
        dwarfdump_cmd = ['llvm-dwarfdump', '--debug-info', elf_file]
    elif subprocess.run(['which', 'readelf'], capture_output=True).returncode == 0:
        dwarfdump_cmd = ['readelf', '--debug-dump=info', elf_file]
    else:
        print("Error: No DWARF dump tool found", file=sys.stderr)
        return {}

    try:
        output = subprocess.check_output(dwarfdump_cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to extract DWARF info: {e}", file=sys.stderr)
        return {}

    kernels = {}

    # Look for kernel argument structures
    # DWARF format has DW_TAG_structure_type on one line, DW_AT_name on another
    lines = output.split('\n')
    i = 0
    while i < len(lines):
        if 'DW_TAG_structure_type' in lines[i]:
            # Check next few lines for DW_AT_name
            for j in range(i+1, min(i+5, len(lines))):
                name_match = re.search(r'DW_AT_name\s+\("(\w+)"\)', lines[j])
                if name_match:
                    struct_name = name_match.group(1)

                    # Only process if it looks like a kernel argument struct
                    if struct_name.endswith('Args') or struct_name.endswith('Params') or 'arg' in struct_name.lower():
                        # Parse the struct to get arguments
                        args = parse_dwarf_struct(output, struct_name, arch)

                        if args:
                            # Derive kernel name from struct name
                            # VecAddArgs -> vecadd
                            kernel_name = struct_name.replace('Args', '').replace('Params', '').replace('_arg_t', '')
                            kernel_name = kernel_name[0].lower() + kernel_name[1:] if kernel_name else 'kernel'

                            kernels[kernel_name] = args
                            print(f"// Found kernel '{kernel_name}' with {len(args)} arguments", file=sys.stderr)
                    break
        i += 1

    return kernels

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
// Generated from DWARF debug info
#include "vortex_hip_runtime.h"
#include <stdio.h>

// Binary data symbols (created by: ld -r -b binary kernel.vxbin)
extern "C" {{
    extern const uint8_t kernel_vxbin[];
    extern const uint8_t kernel_vxbin_end[];
}}

// Calculate binary size
static const size_t kernel_vxbin_size =
    (size_t)(&kernel_vxbin_end[0]) - (size_t)(&kernel_vxbin[0]);

// Kernel handle (set by registration)
void* {kernel_name}_handle = nullptr;

// Metadata array ({len(layout)} arguments)
static const hipKernelArgumentMetadata {kernel_name}_metadata[] = {{
{metadata_array}
}};

// Registration function (called at program startup)
__attribute__((constructor))
static void register_{kernel_name}() {{
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
        fprintf(stderr, "Registered kernel {kernel_name} with %zu bytes binary and {len(layout)} arguments\\n",
                kernel_vxbin_size);
    }}
}}
'''
    return code

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.elf> [options]")
        print(f"")
        print(f"Options:")
        print(f"  --arch=rv32   Force 32-bit RISC-V (default: auto-detect)")
        print(f"  --arch=rv64   Force 64-bit RISC-V")
        sys.exit(1)

    elf_file = sys.argv[1]

    if not os.path.exists(elf_file):
        print(f"Error: File not found: {elf_file}", file=sys.stderr)
        sys.exit(1)

    # Parse options
    arch = None
    for arg in sys.argv[2:]:
        if arg.startswith('--arch='):
            arch = arg.split('=')[1]
            if arch not in ['rv32', 'rv64']:
                print(f"Error: Invalid architecture: {arch}", file=sys.stderr)
                sys.exit(1)

    # Auto-detect if not specified
    if not arch:
        arch = get_arch_from_elf(elf_file)

    print(f"// Extracting metadata from {elf_file} (architecture: {arch})", file=sys.stderr)
    print(f"// Using DWARF debug info parsing", file=sys.stderr)

    # Extract kernel info from DWARF
    kernels = parse_dwarf_info(elf_file, arch)

    if not kernels:
        print(f"// Warning: No HIP kernel argument structures found in {elf_file}", file=sys.stderr)
        print(f"// Make sure the kernel defines an argument structure like 'VecAddArgs'", file=sys.stderr)
        print(f"// and is compiled with debug info (-g flag)", file=sys.stderr)
        sys.exit(0)

    # Generate stub for each kernel
    for kernel_name, args in kernels.items():
        layout = calculate_layout(args)
        code = generate_metadata_code(kernel_name, layout, elf_file)
        print(code)
        print()  # Separator between kernels

if __name__ == '__main__':
    main()
