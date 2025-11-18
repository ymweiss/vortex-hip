#!/usr/bin/env python3
"""
Filter Polygeist GPU kernel variants to keep only first variant per kernel.
This simplifies MLIR output by removing auto-tuning alternatives.

Usage:
    ./filter-gpu-variants.py input.mlir output.mlir
"""

import sys
import re

def extract_base_name(func_name):
    """
    Extract base kernel name by removing Polygeist variant suffix.
    Example: _Z12launch_basicPiS_ji_kernel94565344022848 -> _Z12launch_basicPiS_ji
    """
    # Find _kernel followed by digits
    match = re.search(r'_kernel\d+', func_name)
    if match:
        return func_name[:match.start()]
    return func_name

def filter_variants(input_file, output_file):
    """
    Keep only the first variant of each GPU kernel function.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    in_gpu_module = False
    in_kernel_func = False
    current_func_name = None
    current_func_lines = []
    seen_kernels = set()
    skip_current_func = False
    brace_depth = 0

    for line in lines:
        # Track if we're inside gpu.module
        if 'gpu.module @' in line:
            in_gpu_module = True
            output_lines.append(line)
            continue

        if in_gpu_module:
            # Check for gpu.func with kernel attribute
            if 'gpu.func @' in line and 'kernel' in line:
                in_kernel_func = True
                current_func_lines = [line]
                brace_depth = line.count('{') - line.count('}')

                # Extract function name
                match = re.search(r'@(\w+)', line)
                if match:
                    current_func_name = match.group(1)
                    base_name = extract_base_name(current_func_name)

                    if base_name in seen_kernels:
                        # Skip this variant
                        skip_current_func = True
                    else:
                        # Keep this variant (first one)
                        seen_kernels.add(base_name)
                        skip_current_func = False
                continue

            if in_kernel_func:
                current_func_lines.append(line)
                brace_depth += line.count('{') - line.count('}')

                # Check if function ended
                if brace_depth <= 0:
                    in_kernel_func = False
                    if not skip_current_func:
                        output_lines.extend(current_func_lines)
                    current_func_lines = []
                    skip_current_func = False
                    brace_depth = 0
                continue

            # Check if gpu.module ended
            if line.strip() == '}':
                # Count braces to see if this closes the gpu.module
                # Simple heuristic: if we're not in a function and see a closing brace
                if not in_kernel_func:
                    in_gpu_module = False

        # Pass through all non-kernel lines
        if not in_kernel_func or not skip_current_func:
            output_lines.append(line)

    # Write filtered output
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"Filtered GPU variants:")
    print(f"  Found {len(seen_kernels)} unique kernel(s)")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    return len(seen_kernels)

def main():
    if len(sys.argv) != 3:
        print("Usage: filter-gpu-variants.py input.mlir output.mlir")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        num_kernels = filter_variants(input_file, output_file)
        print(f"✓ Success: Kept first variant of {num_kernels} kernel(s)")
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
