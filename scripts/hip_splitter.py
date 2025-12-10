#!/usr/bin/env python3
"""
hip_splitter.py - Extract kernels from HIP source files

Splits a HIP source file containing both host and kernel code into:
1. Kernel-only files (one per kernel) for Polygeist compilation
2. Host-only file with extern kernel declarations
3. Build script to orchestrate compilation

Usage:
    python hip_splitter.py input.hip -o output_dir/
"""

import re
import sys
import os
import argparse
from pathlib import Path


def extract_kernels(source):
    """
    Extract all __global__ kernel function definitions.

    Returns list of tuples: (kernel_name, full_definition)
    """
    # Pattern to match __global__ function definition with body
    # Handles nested braces by counting depth
    kernels = []

    # Find all __global__ function starts
    kernel_starts = list(re.finditer(
        r'__global__\s+void\s+(\w+)\s*\([^)]*\)\s*\{',
        source
    ))

    for match in kernel_starts:
        kernel_name = match.group(1)
        start_pos = match.start()

        # Find matching closing brace
        brace_count = 0
        in_function = False
        end_pos = start_pos

        for i in range(match.end() - 1, len(source)):
            char = source[i]
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > start_pos:
            kernel_def = source[start_pos:end_pos]
            kernels.append((kernel_name, kernel_def))

    return kernels


def extract_kernel_signature(kernel_def):
    """
    Extract just the function signature from kernel definition.
    Returns: "void kernel_name(args)"
    """
    match = re.search(
        r'__global__\s+(void\s+\w+\s*\([^)]*\))',
        kernel_def
    )
    if match:
        return match.group(1)
    return None


def extract_kernel_launches(source):
    """
    Find all kernel launch sites: kernel<<<grid, block>>>(args)

    Returns list of tuples: (kernel_name, grid_expr, block_expr, args_expr, full_match)
    """
    launches = []

    # Pattern: kernel_name<<<grid, block>>>(args);
    # Note: This is a simplified pattern, may need enhancement for complex expressions
    pattern = r'(\w+)\s*<<<\s*([^,]+),\s*([^>]+)>>>\s*\(([^;]*)\)\s*;'

    for match in re.finditer(pattern, source):
        kernel_name = match.group(1)
        grid_expr = match.group(2).strip()
        block_expr = match.group(3).strip()
        args_expr = match.group(4).strip()
        full_match = match.group(0)

        launches.append((kernel_name, grid_expr, block_expr, args_expr, full_match))

    return launches


def extract_preprocessor_defs(source):
    """
    Extract preprocessor definitions and typedefs that may be needed by kernels.
    Returns string with #define, #ifndef, typedef, etc. before first function.
    Handles multi-line macros (ending with backslash).
    """
    lines = source.split('\n')
    defs = []
    in_multiline_define = False

    for line in lines:
        stripped = line.strip()

        # Stop at first function or kernel definition (but not in middle of multiline)
        if not in_multiline_define:
            if stripped.startswith('__global__') or (stripped.startswith('void ') and '{' in line):
                break

        # Track multi-line macros
        if stripped.startswith('#define'):
            in_multiline_define = stripped.endswith('\\')
            defs.append(line)
        elif in_multiline_define:
            defs.append(line)
            in_multiline_define = stripped.endswith('\\')
        # Include other preprocessor directives and typedefs
        elif (stripped.startswith('#ifndef') or
              stripped.startswith('#endif') or
              stripped.startswith('typedef')):
            defs.append(line)

    return '\n'.join(defs)


def generate_kernel_file(kernel_name, kernel_def, source):
    """
    Generate kernel-only HIP file with launch wrapper.

    Structure:
        #include "hip_runtime_vortex/hip_runtime.h"
        #include <stdint.h>
        [preprocessor definitions from source]

        __global__ void kernel_name(...) { ... }

        void launch_kernel_name(..., int num_blocks, int threads_per_block) {
            kernel_name<<<num_blocks, threads_per_block>>>(...);
        }
    """
    # Extract signature to build launch wrapper
    signature = extract_kernel_signature(kernel_def)

    # Extract parameter list from signature
    params_match = re.search(r'\(([^)]*)\)', signature)
    params = params_match.group(1) if params_match else ""

    # Extract just parameter names for launch call
    param_names = []
    if params.strip():
        for param in params.split(','):
            param = param.strip()
            # Extract variable name (last identifier in declaration)
            name_match = re.search(r'\b(\w+)\s*$', param)
            if name_match:
                param_names.append(name_match.group(1))

    param_call = ', '.join(param_names)

    # Build launch wrapper signature
    launch_params = params
    if launch_params:
        launch_params += ', '
    launch_params += 'int num_blocks, int threads_per_block'

    # Extract preprocessor definitions from original source
    preprocessor_defs = extract_preprocessor_defs(source)

    content = f'''// Kernel extracted from full HIP source
// Auto-generated by hip_splitter.py
#include "hip_runtime_vortex/hip_runtime.h"
#include <stdint.h>

{preprocessor_defs}

{kernel_def}

void launch_{kernel_name}({launch_params}) {{
    {kernel_name}<<<num_blocks, threads_per_block>>>({param_call});
}}
'''

    return content


def generate_host_file(source, kernels, launches):
    """
    Generate host-only C++ file with:
    1. Kernel definitions removed
    2. Extern declarations added
    3. Kernel launches transformed to hipLaunchKernelGGL
    """
    host_source = source

    # Remove kernel definitions
    for kernel_name, kernel_def in kernels:
        host_source = host_source.replace(kernel_def, '')

    # Extract kernel signatures for extern declarations
    extern_decls = []
    for kernel_name, kernel_def in kernels:
        signature = extract_kernel_signature(kernel_def)
        if signature:
            # Convert to extern "C" declaration
            extern_decl = f'extern "C" {signature};'
            extern_decls.append(extern_decl)

    # Add extern declarations at the top (after includes)
    # Find the last #include
    include_matches = list(re.finditer(r'#include\s+[<"][^>"]+[>"]', host_source))
    if include_matches:
        insert_pos = include_matches[-1].end()

        # Build extern declarations block
        extern_block = '\n\n// Kernel declarations (definitions compiled separately)\n'
        extern_block += '\n'.join(extern_decls) + '\n'

        host_source = host_source[:insert_pos] + extern_block + host_source[insert_pos:]

    # Transform kernel launches
    # kernel<<<grid, block>>>(args); → hipLaunchKernelGGL(kernel, grid, block, 0, 0, args);
    for kernel_name, grid_expr, block_expr, args_expr, full_match in launches:
        # Build replacement
        replacement = f'hipLaunchKernelGGL({kernel_name}, {grid_expr}, {block_expr}, 0, 0, {args_expr});'
        host_source = host_source.replace(full_match, replacement)

    return host_source


def generate_build_script(base_name, kernel_names, output_dir):
    """
    Generate build script that:
    1. Compiles each kernel through Polygeist pipeline
    2. Extracts metadata from Vortex MLIR
    3. Compiles host code
    4. Links everything together
    """
    script = f'''#!/bin/bash
# Auto-generated build script by hip_splitter.py
# Compiles {base_name} using split HIP compilation pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Building {base_name}..."

'''

    # Add kernel compilation steps
    for kernel_name in kernel_names:
        script += f'''
echo "Compiling kernel: {kernel_name}..."

# Stage 1: HIP → GPU dialect MLIR
$REPO_ROOT/scripts/polygeist/hip-to-gpu-dialect.sh \\
    $SCRIPT_DIR/{kernel_name}_kernel.hip

# Stage 2: GPU dialect → Vortex MLIR
$REPO_ROOT/Polygeist/build/bin/polygeist-opt \\
    --reorder-gpu-kernel-args \\
    --convert-gpu-to-vortex \\
    $SCRIPT_DIR/{kernel_name}_kernel.mlir \\
    -o $SCRIPT_DIR/{kernel_name}_kernel_vortex.mlir

# Stage 3: Extract metadata
python3 $REPO_ROOT/scripts/extract_kernel_metadata.py \\
    $SCRIPT_DIR/{kernel_name}_kernel_vortex.mlir \\
    > $SCRIPT_DIR/{kernel_name}_metadata.h

# Stage 4: MLIR → LLVM IR
$REPO_ROOT/Polygeist/build/bin/mlir-translate \\
    --mlir-to-llvmir \\
    $SCRIPT_DIR/{kernel_name}_kernel_vortex.mlir \\
    -o $SCRIPT_DIR/{kernel_name}_kernel.ll

# Stage 5: LLVM IR → RISC-V binary
$REPO_ROOT/Polygeist/build/llvm-project/build/bin/llc \\
    -march=riscv32 \\
    -mattr=+m,+a,+f,+d \\
    $SCRIPT_DIR/{kernel_name}_kernel.ll \\
    -o $SCRIPT_DIR/{kernel_name}_kernel.s

# TODO: Assemble and embed kernel binary
'''

    # Add host compilation
    script += f'''
echo "Compiling host code..."
# TODO: Implement host compilation once runtime library is ready
# clang++ {base_name}_host.cpp -lhip_vortex_runtime -lvortex -o {base_name}

echo "Build complete!"
echo "Kernel binaries: {' '.join([f'{k}_kernel.s' for k in kernel_names])}"
'''

    return script


def main():
    parser = argparse.ArgumentParser(
        description='Extract kernels from HIP source file'
    )
    parser.add_argument('input', help='Input HIP source file')
    parser.add_argument('-o', '--output', help='Output directory', default='.')

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input file
    with open(input_file, 'r') as f:
        source = f.read()

    # Extract kernels
    kernels = extract_kernels(source)

    if not kernels:
        print(f"Warning: No kernels found in {input_file}", file=sys.stderr)
        return 0

    print(f"Found {len(kernels)} kernel(s): {', '.join([k[0] for k in kernels])}")

    # Generate kernel-only files
    kernel_names = []
    for kernel_name, kernel_def in kernels:
        kernel_names.append(kernel_name)
        kernel_content = generate_kernel_file(kernel_name, kernel_def, source)

        output_file = output_dir / f'{kernel_name}_kernel.hip'
        with open(output_file, 'w') as f:
            f.write(kernel_content)
        print(f"  Generated: {output_file}")

    # Extract kernel launches
    launches = extract_kernel_launches(source)
    print(f"Found {len(launches)} kernel launch(es)")

    # Generate host file
    host_content = generate_host_file(source, kernels, launches)
    base_name = input_file.stem
    host_file = output_dir / f'{base_name}_host.cpp'
    with open(host_file, 'w') as f:
        f.write(host_content)
    print(f"  Generated: {host_file}")

    # Generate build script
    build_script = generate_build_script(base_name, kernel_names, output_dir)
    script_file = output_dir / f'build_{base_name}.sh'
    with open(script_file, 'w') as f:
        f.write(build_script)
    script_file.chmod(0o755)
    print(f"  Generated: {script_file}")

    print("\nDone! Run build script to compile:")
    print(f"  {script_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
