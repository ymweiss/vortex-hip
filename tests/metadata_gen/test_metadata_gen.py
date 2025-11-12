#!/usr/bin/env python3
"""
Unit tests for HIP metadata generator script
Tests the detection and extraction of kernel metadata from ELF files
"""

import unittest
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# Add Vortex scripts directory to path
SCRIPT_DIR = Path(__file__).parent
VORTEX_ROOT = Path.home() / "vortex"
sys.path.insert(0, str(VORTEX_ROOT / "scripts"))

# Import the metadata generator functions
try:
    import hip_metadata_gen as gen
except ImportError:
    print("Error: Cannot import hip_metadata_gen", file=sys.stderr)
    print(f"Make sure script exists at {VORTEX_ROOT}/vortex/scripts/hip_metadata_gen.py", file=sys.stderr)
    sys.exit(1)


class TestArgumentLayout(unittest.TestCase):
    """Test argument layout calculation with proper alignment"""

    def test_padding_calculation_aligned(self):
        """Test already aligned offset needs no padding"""
        args = [gen.ArgInfo(name='a', size=4, alignment=4, is_pointer=True)]
        layout = gen.calculate_layout(args)

        self.assertEqual(1, len(layout))
        self.assertEqual(0, layout[0]['offset'])
        self.assertEqual(4, layout[0]['size'])

    def test_padding_calculation_unaligned(self):
        """Test unaligned offset gets proper padding"""
        args = [
            gen.ArgInfo(name='c', size=1, alignment=1, is_pointer=False),  # char at 0
            gen.ArgInfo(name='i', size=4, alignment=4, is_pointer=False)   # int needs padding
        ]
        layout = gen.calculate_layout(args)

        self.assertEqual(2, len(layout))
        self.assertEqual(0, layout[0]['offset'])  # char at 0
        self.assertEqual(4, layout[1]['offset'])  # int at 4 (3 bytes padding)

    def test_rv32_pointer_layout(self):
        """Test RV32: float* a, float* b, int n"""
        args = [
            gen.ArgInfo(name='a', size=4, alignment=4, is_pointer=True),
            gen.ArgInfo(name='b', size=4, alignment=4, is_pointer=True),
            gen.ArgInfo(name='n', size=4, alignment=4, is_pointer=False)
        ]
        layout = gen.calculate_layout(args)

        self.assertEqual(3, len(layout))
        self.assertEqual(0, layout[0]['offset'])
        self.assertEqual(4, layout[1]['offset'])
        self.assertEqual(8, layout[2]['offset'])

    def test_rv64_pointer_layout(self):
        """Test RV64: double* a, double* b, int n"""
        args = [
            gen.ArgInfo(name='a', size=8, alignment=8, is_pointer=True),
            gen.ArgInfo(name='b', size=8, alignment=8, is_pointer=True),
            gen.ArgInfo(name='n', size=4, alignment=4, is_pointer=False)
        ]
        layout = gen.calculate_layout(args)

        self.assertEqual(3, len(layout))
        self.assertEqual(0, layout[0]['offset'])
        self.assertEqual(8, layout[1]['offset'])
        self.assertEqual(16, layout[2]['offset'])

    def test_mixed_types_with_padding(self):
        """Test: char c, double* ptr (RV64)"""
        args = [
            gen.ArgInfo(name='c', size=1, alignment=1, is_pointer=False),
            gen.ArgInfo(name='ptr', size=8, alignment=8, is_pointer=True)
        ]
        layout = gen.calculate_layout(args)

        self.assertEqual(2, len(layout))
        self.assertEqual(0, layout[0]['offset'])  # char at 0
        self.assertEqual(8, layout[1]['offset'])  # ptr at 8 (7 bytes padding)


class TestTypeSizes(unittest.TestCase):
    """Test type size mappings for RV32 and RV64"""

    def test_rv32_pointer_size(self):
        """RV32 pointers are 4 bytes"""
        size, align, is_ptr = gen.RV32_TYPE_INFO['pointer']
        self.assertEqual(4, size)
        self.assertEqual(4, align)
        self.assertTrue(is_ptr)

    def test_rv64_pointer_size(self):
        """RV64 pointers are 8 bytes"""
        size, align, is_ptr = gen.RV64_TYPE_INFO['pointer']
        self.assertEqual(8, size)
        self.assertEqual(8, align)
        self.assertTrue(is_ptr)

    def test_int_size_consistent(self):
        """int is 4 bytes on both RV32 and RV64"""
        rv32_int = gen.RV32_TYPE_INFO['int']
        rv64_int = gen.RV64_TYPE_INFO['int']
        self.assertEqual(4, rv32_int[0])
        self.assertEqual(4, rv64_int[0])

    def test_long_size_differs(self):
        """long differs between RV32 (4) and RV64 (8)"""
        rv32_long = gen.RV32_TYPE_INFO['long']
        rv64_long = gen.RV64_TYPE_INFO['long']
        self.assertEqual(4, rv32_long[0])
        self.assertEqual(8, rv64_long[0])


class TestMetadataCodeGeneration(unittest.TestCase):
    """Test C++ code generation from metadata"""

    def test_generates_valid_metadata_array(self):
        """Generated metadata array should be valid C++"""
        layout = [
            {'name': 'a', 'offset': 0, 'size': 4, 'alignment': 4, 'is_pointer': 1},
            {'name': 'n', 'offset': 4, 'size': 4, 'alignment': 4, 'is_pointer': 0}
        ]

        code = gen.generate_metadata_code('test_kernel', layout, 'test.vxbin')

        # Check essential elements
        self.assertIn('hipKernelArgumentMetadata', code)
        self.assertIn('test_kernel_metadata', code)
        self.assertIn('.offset = 0', code)
        self.assertIn('.offset = 4', code)
        self.assertIn('.is_pointer = 1', code)
        self.assertIn('.is_pointer = 0', code)
        self.assertIn('__attribute__((constructor))', code)
        self.assertIn('__hipRegisterFunctionWithMetadata', code)

    def test_handles_empty_layout(self):
        """Empty layout should return empty string"""
        code = gen.generate_metadata_code('empty_kernel', [], 'test.vxbin')
        self.assertEqual("", code)

    def test_includes_kernel_handle(self):
        """Generated code should include kernel handle variable"""
        layout = [
            {'name': 'a', 'offset': 0, 'size': 4, 'alignment': 4, 'is_pointer': 1}
        ]

        code = gen.generate_metadata_code('my_kernel', layout, 'test.vxbin')

        self.assertIn('void* my_kernel_handle', code)


class TestArchDetection(unittest.TestCase):
    """Test ELF architecture detection"""

    def test_default_arch_rv64(self):
        """Should default to rv64 if detection fails"""
        # Test with non-existent file
        arch = gen.get_arch_from_elf('/nonexistent/file.elf')
        self.assertEqual('rv64', arch)


class TestDWARFParsing(unittest.TestCase):
    """Test DWARF parsing (integration tests - requires real ELF)"""

    @unittest.skipUnless(os.path.exists('/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'),
                        "Requires vecadd kernel.elf")
    def test_parse_vecadd_kernel_detects_arch(self):
        """Test that vecadd kernel is detected as RV32"""
        elf_path = '/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'

        # Should detect RV32
        arch = gen.get_arch_from_elf(elf_path)
        self.assertEqual('rv32', arch)

    @unittest.skipUnless(os.path.exists('/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'),
                        "Requires vecadd kernel.elf")
    def test_parse_vecadd_kernel_finds_kernel_body(self):
        """Test that kernel_body function is detected"""
        elf_path = '/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'
        arch = gen.get_arch_from_elf(elf_path)

        # Should find kernel_body function
        kernels = gen.parse_dwarf_info(elf_path, arch)

        # This should pass after implementing proper detection
        self.assertIn('kernel_body', kernels, "kernel_body function should be detected")
        self.assertIsNotNone(kernels.get('kernel_body'))

    @unittest.skipUnless(os.path.exists('/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'),
                        "Requires vecadd kernel.elf")
    def test_parse_vecadd_extracts_correct_args(self):
        """Test that kernel_body arguments are extracted correctly"""
        elf_path = '/home/yaakov/vortex_hip/tests/vecadd_metadata_test/kernel.elf'
        arch = gen.get_arch_from_elf(elf_path)

        kernels = gen.parse_dwarf_info(elf_path, arch)

        # kernel_body should have 4 arguments: a, b, c, n
        # After the runtime fields (grid_dim, block_dim, shared_mem)
        if 'kernel_body' in kernels:
            args = kernels['kernel_body']
            self.assertEqual(4, len(args), "Should have 4 kernel arguments")

            # RV32: pointers are 4 bytes
            self.assertEqual(4, args[0].size, "First arg (float*) should be 4 bytes")
            self.assertTrue(args[0].is_pointer, "First arg should be pointer")

            self.assertEqual(4, args[1].size, "Second arg (float*) should be 4 bytes")
            self.assertTrue(args[1].is_pointer, "Second arg should be pointer")

            self.assertEqual(4, args[2].size, "Third arg (float*) should be 4 bytes")
            self.assertTrue(args[2].is_pointer, "Third arg should be pointer")

            self.assertEqual(4, args[3].size, "Fourth arg (uint32_t) should be 4 bytes")
            self.assertFalse(args[3].is_pointer, "Fourth arg should not be pointer")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
