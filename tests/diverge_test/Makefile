# Makefile for HIP diverge Test with Metadata
# Adapted from basic_test
# Tests vector addition with varying sizes and multi-block execution

# =============================================================================
# Configuration
# =============================================================================

# Use relative paths to vortex_hip project structure
VORTEX_HIP_HOME ?= $(shell cd ../.. && pwd)
VORTEX_HOME ?= $(VORTEX_HIP_HOME)/vortex

# Check if Vortex is built
VORTEX_BUILD_CONFIG := $(VORTEX_HOME)/build/config.mk
ifeq (,$(wildcard $(VORTEX_BUILD_CONFIG)))
    $(error Vortex not built. Please run: cd $(VORTEX_HOME) && make)
endif

# Load Vortex configuration
include $(VORTEX_BUILD_CONFIG)

# Metadata generator script
METADATA_GEN = $(VORTEX_HOME)/scripts/hip_metadata_gen.py

# =============================================================================
# Vortex Kernel Build Configuration
# =============================================================================

VORTEX_RT_PATH ?= $(VORTEX_HOME)/runtime
VORTEX_KN_PATH ?= $(VORTEX_HOME)/build/kernel

# XLEN should come from config.mk, but default to 32 if not set
XLEN ?= 32

# Architecture flags
ifeq ($(XLEN),64)
	VX_CFLAGS += -march=rv64imafd -mabi=lp64d
	STARTUP_ADDR ?= 0x180000000
else
	VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
	STARTUP_ADDR ?= 0x80000000
endif

# LLVM flags (if LLVM_VORTEX is set)
ifneq (,$(LLVM_VORTEX))
	LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
	LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
	LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
	LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
	LLVM_CFLAGS += -mllvm -disable-loop-idiom-all

	VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
	VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
	VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump
	VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy
else
	$(error LLVM_VORTEX not set. Please source $(VORTEX_HOME)/ci/toolchain_env.sh)
endif

# Kernel compilation flags
VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions
VX_CFLAGS += -nostartfiles -nostdlib
VX_CFLAGS += -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include
VX_CFLAGS += -I$(VORTEX_HOME)/hw
VX_CFLAGS += -I$(VORTEX_HOME)/build/hw
VX_CFLAGS += -DXLEN_$(XLEN) -DNDEBUG

# IMPORTANT: Add -g for debug info (needed for metadata extraction)
VX_CFLAGS += -g

# Kernel libraries
VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc
VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

# Kernel linker flags
VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR)
VX_LDFLAGS += $(VORTEX_KN_PATH)/libvortex.a $(VX_LIBS)

# =============================================================================
# Host Build Configuration
# =============================================================================

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic
CXXFLAGS += -I$(VORTEX_HIP_HOME)/runtime/include
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include

LDFLAGS += -L$(VORTEX_HIP_HOME)/runtime/build -lhip_vortex
LDFLAGS += -Wl,-rpath,$(VORTEX_HIP_HOME)/runtime/build

ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean run test info show-metadata

all: diverge_test

# === Phase 1: Compile kernel to ELF (with debug info) ===
kernel.elf: kernel.cpp
	@echo "=== [1/6] Compiling kernel to ELF (with debug info) ==="
	@echo "  Compiler: $(VX_CXX)"
	@echo "  Flags: $(VX_CFLAGS)"
	$(VX_CXX) $(VX_CFLAGS) $< $(VX_LDFLAGS) -o $@
	@echo ""

# === Phase 2: Generate metadata from DWARF debug info (Python script) ===
kernel_metadata.cpp: kernel.elf
	@echo "=== [2/6] Generating metadata from DWARF debug info ==="
	@echo "  This is Phase 1 (Python prototype) metadata generation"
	@if [ ! -f "$(METADATA_GEN)" ]; then \
		echo "Error: Metadata generator not found at $(METADATA_GEN)"; \
		exit 1; \
	fi
	@echo "  Running: python3 $(METADATA_GEN) kernel.elf > kernel_metadata.cpp"
	python3 $(METADATA_GEN) $< > $@
	@echo "  Generated: $@"
	@echo ""

# === Phase 3: Compile metadata stub ===
kernel_metadata.o: kernel_metadata.cpp
	@echo "=== [3/6] Compiling metadata registration stub ==="
	$(CXX) -c $(CXXFLAGS) $< -o $@
	@echo ""

# === Phase 4: Convert kernel to Vortex binary format ===
kernel.vxbin: kernel.elf
	@echo "=== [4/6] Converting kernel to Vortex binary format ==="
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@
	@echo ""

# === Phase 5: Create binary object file ===
kernel_vxbin.o: kernel.vxbin
	@echo "=== [5/6] Creating binary object file ==="
	@echo "  Embedding binary as .rodata section"
	$(LD) -r -b binary -o $@ $<
	objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents \
		--redefine-sym _binary_kernel_vxbin_start=kernel_vxbin \
		--redefine-sym _binary_kernel_vxbin_end=kernel_vxbin_end \
		$@
	@echo ""

# === Phase 6: Link host application with metadata stub and kernel binary ===
diverge_test: main.o kernel_metadata.o kernel_vxbin.o
	@echo "=== [6/6] Linking final application ==="
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
	@echo ""
	@echo "=== Build Complete ==="
	@echo ""
	@echo "Run with: ./diverge_test [-n size]"
	@echo "Example:  ./diverge_test -n 256"
	@echo ""

main.o: main.cpp
	@echo "  Compiling main.cpp"
	$(CXX) -c $(CXXFLAGS) $< -o $@

# Generate disassembly for debugging
kernel.dump: kernel.elf
	$(VX_DP) -D $< > $@

# =============================================================================
# Utility Targets
# =============================================================================

# Info target
info:
	@echo "=== diverge Test Configuration ==="
	@echo ""
	@echo "Vortex Configuration:"
	@echo "  VORTEX_HOME:     $(VORTEX_HOME)"
	@echo "  VORTEX_HIP_HOME: $(VORTEX_HIP_HOME)"
	@echo "  XLEN:            $(XLEN)"
	@echo "  LLVM_VORTEX:     $(LLVM_VORTEX)"
	@echo ""
	@echo "Metadata Generation:"
	@echo "  Script: $(METADATA_GEN)"
	@echo "  Phase:  1 (Python prototype)"
	@echo ""
	@echo "Build Flow:"
	@echo "  kernel.cpp -> kernel.elf (with -g debug info)"
	@echo "           ├-> kernel.vxbin (device binary)"
	@echo "           └-> kernel_metadata.cpp (generated stub)"
	@echo "                   ↓"
	@echo "  main.cpp + kernel_metadata.o + kernel_vxbin.o -> diverge_test"
	@echo ""

# Test target - run with various sizes
test: diverge_test
	@echo "=== Running diverge Test Suite ==="
	@echo ""
	@echo "Test 1: Small (16 elements)"
	./diverge_test -n 16
	@echo ""
	@echo "Test 2: Medium (256 elements)"
	./diverge_test -n 256
	@echo ""
	@echo "Test 3: Large (1024 elements)"
	./diverge_test -n 1024
	@echo ""
	@echo "=== All Tests Complete ==="

# Run target - single default test
run: diverge_test
	./diverge_test -n 16

# Show generated metadata
show-metadata: kernel_metadata.cpp
	@echo "=== Generated Metadata ==="
	@cat $<

clean:
	rm -f *.o *.elf *.vxbin *.dump kernel_metadata.cpp diverge_test
	@echo "Cleaned build artifacts"
