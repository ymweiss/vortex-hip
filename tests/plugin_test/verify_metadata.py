#!/usr/bin/env python3
"""
Verify HIPMetadataExtractor plugin generates correct metadata offsets.

Expected offsets for vecadd(float* a, float* b, float* c, int n):
  Runtime fields: 32 bytes (grid_dim[12] + block_dim[12] + shared_mem[8])
  a: offset 32, size 4, align 4, is_pointer 1
  b: offset 36, size 4, align 4, is_pointer 1
  c: offset 40, size 4, align 4, is_pointer 1
  n: offset 44, size 4, align 4, is_pointer 0

Expected offsets for mixed_types with alignment:
  Runtime fields: 32 bytes
  char c:     offset 32, size 1, align 1
  short s:    offset 34, size 2, align 2  (aligned from 33)
  int i:      offset 36, size 4, align 4
  long l:     offset 40, size 8, align 8  (already aligned)
  float* ptr1: offset 48, size 4, align 4
  double* ptr2: offset 52, size 4, align 4
  int final:   offset 56, size 4, align 4
"""

import sys
import re

def parse_metadata(filename):
    """Parse generated metadata file and extract argument info."""
    with open(filename, 'r') as f:
        content = f.read()

    kernels = {}

    # Find all metadata arrays
    pattern = r'static const struct hipKernelArgumentMetadata (\w+)_metadata\[\] = \{([^}]+)\};'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

    for match in matches:
        kernel_name = match.group(1)
        metadata_body = match.group(2)

        # Parse each argument line: {offset, size, alignment, is_pointer}, // name : type
        arg_pattern = r'\{(\d+), (\d+), (\d+), ([01])\}, // (\w+) : (.+)'
        args = []

        for arg_match in re.finditer(arg_pattern, metadata_body):
            args.append({
                'offset': int(arg_match.group(1)),
                'size': int(arg_match.group(2)),
                'alignment': int(arg_match.group(3)),
                'is_pointer': int(arg_match.group(4)),
                'name': arg_match.group(5),
                'type': arg_match.group(6).strip()
            })

        kernels[kernel_name] = args

    return kernels

def verify_vecadd(args):
    """Verify vecadd kernel metadata."""
    expected = [
        {'name': 'a', 'offset': 32, 'size': 4, 'alignment': 4, 'is_pointer': 1},
        {'name': 'b', 'offset': 36, 'size': 4, 'alignment': 4, 'is_pointer': 1},
        {'name': 'c', 'offset': 40, 'size': 4, 'alignment': 4, 'is_pointer': 1},
        {'name': 'n', 'offset': 44, 'size': 4, 'alignment': 4, 'is_pointer': 0},
    ]

    if len(args) != len(expected):
        print(f"❌ vecadd: Expected {len(expected)} args, got {len(args)}")
        return False

    all_correct = True
    for i, (exp, actual) in enumerate(zip(expected, args)):
        match = (exp['name'] == actual['name'] and
                exp['offset'] == actual['offset'] and
                exp['size'] == actual['size'] and
                exp['alignment'] == actual['alignment'] and
                exp['is_pointer'] == actual['is_pointer'])

        if not match:
            print(f"❌ vecadd arg {i} ({exp['name']}):")
            print(f"   Expected: offset={exp['offset']}, size={exp['size']}, "
                  f"align={exp['alignment']}, ptr={exp['is_pointer']}")
            print(f"   Got:      offset={actual['offset']}, size={actual['size']}, "
                  f"align={actual['alignment']}, ptr={actual['is_pointer']}")
            all_correct = False
        else:
            print(f"✅ vecadd arg {i} ({exp['name']}): offset={actual['offset']}, "
                  f"size={actual['size']}, align={actual['alignment']}, ptr={actual['is_pointer']}")

    return all_correct

def verify_mixed_types(args):
    """Verify mixed_types kernel metadata with alignment."""
    expected = [
        {'name': 'c', 'offset': 32, 'size': 1, 'alignment': 1, 'is_pointer': 0},
        {'name': 's', 'offset': 34, 'size': 2, 'alignment': 2, 'is_pointer': 0},  # aligned
        {'name': 'i', 'offset': 36, 'size': 4, 'alignment': 4, 'is_pointer': 0},
        {'name': 'l', 'offset': 40, 'size': 8, 'alignment': 8, 'is_pointer': 0},
        {'name': 'ptr1', 'offset': 48, 'size': 4, 'alignment': 4, 'is_pointer': 1},
        {'name': 'ptr2', 'offset': 52, 'size': 4, 'alignment': 4, 'is_pointer': 1},
        {'name': 'final', 'offset': 56, 'size': 4, 'alignment': 4, 'is_pointer': 0},
    ]

    if len(args) != len(expected):
        print(f"❌ mixed_types: Expected {len(expected)} args, got {len(args)}")
        return False

    all_correct = True
    for i, (exp, actual) in enumerate(zip(expected, args)):
        match = (exp['name'] == actual['name'] and
                exp['offset'] == actual['offset'] and
                exp['size'] == actual['size'] and
                exp['alignment'] == actual['alignment'] and
                exp['is_pointer'] == actual['is_pointer'])

        if not match:
            print(f"❌ mixed_types arg {i} ({exp['name']}):")
            print(f"   Expected: offset={exp['offset']}, size={exp['size']}, "
                  f"align={exp['alignment']}, ptr={exp['is_pointer']}")
            print(f"   Got:      offset={actual['offset']}, size={actual['size']}, "
                  f"align={actual['alignment']}, ptr={actual['is_pointer']}")
            all_correct = False
        else:
            print(f"✅ mixed_types arg {i} ({exp['name']}): offset={actual['offset']}, "
                  f"size={actual['size']}, align={actual['alignment']}, ptr={actual['is_pointer']}")

    return all_correct

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_metadata.py <metadata_file>")
        sys.exit(1)

    metadata_file = sys.argv[1]

    print("=" * 70)
    print("HIPMetadataExtractor Plugin Verification")
    print("=" * 70)
    print()

    try:
        kernels = parse_metadata(metadata_file)
    except Exception as e:
        print(f"❌ Failed to parse metadata file: {e}")
        sys.exit(1)

    print(f"Found {len(kernels)} kernel(s): {', '.join(kernels.keys())}")
    print()

    all_pass = True

    # Verify vecadd
    if 'vecadd' in kernels:
        print("Testing vecadd kernel:")
        print("-" * 70)
        if not verify_vecadd(kernels['vecadd']):
            all_pass = False
        print()
    else:
        print("❌ vecadd kernel not found")
        all_pass = False

    # Verify mixed_types
    if 'mixed_types' in kernels:
        print("Testing mixed_types kernel:")
        print("-" * 70)
        if not verify_mixed_types(kernels['mixed_types']):
            all_pass = False
        print()
    else:
        print("❌ mixed_types kernel not found")
        all_pass = False

    print("=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED - Plugin generates correct metadata!")
        print()
        print("Key verification:")
        print("  • User arguments start at offset 32 (not 0!)")
        print("  • Alignment is correctly handled")
        print("  • Pointer types are correctly identified")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Plugin has issues")
        sys.exit(1)

if __name__ == '__main__':
    main()
