#!/usr/bin/env python3
"""
inject_kernel_launchers.py - Prepare HIP kernel source for Polygeist compilation

This script processes HIP source files to prepare them for device compilation
through Polygeist. It performs two transformations:

1. Replaces standard HIP includes with hip_runtime_vortex/hip_runtime.h which
   provides __clang_cuda_builtin_vars.h for proper GPU variable support

2. Adds synthetic launch wrappers for each __global__ kernel function.
   This is required because Polygeist needs <<<>>> launch syntax to trigger
   GPU dialect generation.

Usage:
    python inject_kernel_launchers.py input.hip output.cu
    python inject_kernel_launchers.py input.hip  # outputs to stdout

After processing, compile the output with Polygeist using:

    cgeist output.cu \\
        --cuda-lower \\
        --emit-cuda \\
        --cuda-gpu-arch=sm_60 \\
        -nocudalib \\
        -nocudainc \\
        -resource-dir=$POLYGEIST/llvm-project/build/lib/clang/18 \\
        -I$VORTEX_HIP \\
        --function='*' \\
        --output-intermediate-gpu=1 \\
        -S \\
        -o output.mlir

Where:
    $POLYGEIST = path to Polygeist source directory
    $VORTEX_HIP = path to vortex_hip repository root (contains hip_runtime_vortex/)
"""

import re
import sys


def extract_kernels(source: str) -> list:
    """
    Extract kernel function declarations from HIP/CUDA source.
    Returns list of tuples: (kernel_name, params_str, param_names)
    """
    # Pattern to match __global__ function declarations
    # Handles: __global__ void name(params) and __global__ __attribute__(...) void name(params)
    kernel_pattern = re.compile(
        r'__global__\s+'
        r'(?:__attribute__\s*\(\([^)]*\)\)\s*)?'  # Optional __attribute__((...))
        r'(?:__launch_bounds__\s*\([^)]*\)\s*)?'  # Optional __launch_bounds__(...)
        r'void\s+'
        r'(\w+)\s*'                               # Kernel name (captured)
        r'\(([^)]*)\)',                           # Parameters (captured)
        re.MULTILINE
    )

    kernels = []
    for match in kernel_pattern.finditer(source):
        kernel_name = match.group(1)
        params_str = match.group(2).strip()

        # Parse parameter names from the params string
        param_names = []
        if params_str:
            # Split by comma, extract the variable name (last word before any default value)
            for param in params_str.split(','):
                param = param.strip()
                # Remove array brackets like "float data[]" -> "float data"
                param = re.sub(r'\[[^\]]*\]', '', param)
                # Remove * and & from the parameter
                param = param.replace('*', ' ').replace('&', ' ')
                # Get the last word (variable name)
                words = param.split()
                if words:
                    # Skip 'const' and type qualifiers
                    name = words[-1]
                    if name and name not in ('const', 'volatile', 'restrict'):
                        param_names.append(name)

        kernels.append((kernel_name, params_str, param_names))

    return kernels


def generate_launcher(kernel_name: str, params_str: str, param_names: list) -> str:
    """Generate a synthetic launch wrapper for a kernel."""
    args = ', '.join(param_names)

    # Create wrapper function name
    wrapper_name = f"__polygeist_launch_{kernel_name}"

    # Add grid/block dimension parameters
    extended_params = f"{params_str}, int __blocks, int __threads"

    return f'''
// Synthetic launch wrapper for {kernel_name} (required for Polygeist GPU codegen)
extern "C" __attribute__((used))
void {wrapper_name}({extended_params}) {{
    {kernel_name}<<<__blocks, __threads>>>({args});
}}
'''


def transform_includes(source: str) -> str:
    """Transform HIP includes for dual host/device compilation.

    Creates conditional include using __CUDACC__ (defined by clang CUDA frontend):
    - Device compilation (__CUDACC__ defined): hip_runtime_vortex/hip_runtime.h
    - Host compilation (__CUDACC__ not defined): hip/hip_runtime.h (Vortex runtime)
    """
    # Pattern to match various forms of hip/hip_runtime.h include
    include_patterns = [
        r'#include\s*<hip/hip_runtime\.h>',
        r'#include\s*"hip/hip_runtime\.h"',
        r'#include\s*<hip_runtime\.h>',
        r'#include\s*"hip_runtime\.h"',
    ]

    replacement = '''#ifndef __CUDA__
#include "hip/hip_runtime.h"  // Host compilation (Vortex runtime)
#else
#include "hip_runtime_vortex/hip_runtime.h"  // Device compilation (Polygeist)
#endif'''

    result = source
    for pattern in include_patterns:
        if re.search(pattern, result):
            result = re.sub(pattern, replacement, result, count=1)
            # Only replace once, to avoid duplicate conditional blocks
            break

    return result


def convert_kernel_to_declaration(source: str) -> str:
    """Convert kernel function definitions to forward declarations for host compilation.

    Wraps each kernel body with #ifndef __VORTEX_HOST__ / #endif so that:
    - Host compilation (with -D__VORTEX_HOST__): kernels become forward declarations
    - Device compilation: kernels have full bodies
    """
    # Pattern to match __global__ function with body
    # Captures: full signature, then finds matching braces for body
    kernel_def_pattern = re.compile(
        r'(__global__\s+'
        r'(?:__attribute__\s*\(\([^)]*\)\)\s*)?'
        r'(?:__launch_bounds__\s*\([^)]*\)\s*)?'
        r'void\s+\w+\s*\([^)]*\))\s*'
        r'\{',
        re.MULTILINE
    )

    result = source
    offset = 0

    for match in kernel_def_pattern.finditer(source):
        signature = match.group(1)
        body_start = match.end() - 1  # Position of opening brace

        # Find matching closing brace
        brace_count = 1
        pos = body_start + 1
        while brace_count > 0 and pos < len(source):
            if source[pos] == '{':
                brace_count += 1
            elif source[pos] == '}':
                brace_count -= 1
            pos += 1

        body_end = pos  # Position after closing brace
        body = source[body_start:body_end]

        # Create replacement with conditional compilation
        replacement = f'''{signature}
#ifdef __VORTEX_HOST__
;  // Forward declaration for host compilation
#else
{body}
#endif'''

        # Apply replacement with offset adjustment
        result = result[:body_start + offset - len(signature) - len(match.group(0)) + len(signature) + 1] + \
                 replacement + \
                 result[body_end + offset:]

        # This is getting complex, let's use a simpler approach
        break  # We'll implement a simpler version

    # Simpler approach: replace entire kernel functions
    return _convert_kernels_simple(source)


def _convert_kernels_simple(source: str) -> str:
    """Simple kernel conversion using regex substitution."""
    # This pattern matches kernel function definitions
    # We'll wrap the body in conditional compilation

    def replace_kernel(match):
        full_match = match.group(0)
        signature = match.group(1)
        body = match.group(2)

        # For host compilation, convert to extern declaration
        # Remove __global__ and add extern for host
        host_signature = re.sub(r'__global__\s*', '', signature)

        return f'''#ifndef __CUDA__
extern {host_signature};  // Extern declaration for host compilation
#else
{signature}
{{{body}}}
#endif'''

    # Pattern: __global__ ... void name(params) { body }
    # Need to handle nested braces in body
    pattern = re.compile(
        r'(__global__\s+'
        r'(?:__attribute__\s*\(\([^)]*\)\)\s*)?'
        r'(?:__launch_bounds__\s*\([^)]*\)\s*)?'
        r'void\s+\w+\s*\([^)]*\))\s*'
        r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    return pattern.sub(replace_kernel, source)


def inject_launchers(source: str, for_device: bool = True) -> str:
    """Inject synthetic launch wrappers into the source code.

    Args:
        source: Source code to process
        for_device: If True, prepare for device compilation (Polygeist)
                   If False, prepare for host compilation (standard C++)
    """
    kernels = extract_kernels(source)

    if not kernels:
        return source

    if for_device:
        # Transform includes for Polygeist compatibility
        result = transform_includes(source)
    else:
        # For host compilation, keep original includes
        result = source

    # Convert kernel definitions to conditional declarations
    result = _convert_kernels_simple(result)

    if for_device:
        # Generate all launchers for device compilation
        launchers = []
        for kernel_name, params_str, param_names in kernels:
            launcher = generate_launcher(kernel_name, params_str, param_names)
            launchers.append(launcher)

        # Add launchers at the end of the file (only for device path)
        launcher_block = '''
// ============================================================================
// Polygeist synthetic launch wrappers (auto-generated)
// These wrappers are required for Polygeist to generate GPU dialect code
// when compiling kernels without explicit launch calls.
// ============================================================================
#ifdef __CUDA__
''' + ''.join(launchers) + '''
#endif
'''
        result = result + launcher_block

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_file, 'r') as f:
        source = f.read()

    # Generate output suitable for both host and device compilation
    result = inject_launchers(source, for_device=True)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
