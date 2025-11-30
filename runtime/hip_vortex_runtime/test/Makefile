# Makefile for HIP Vortex Runtime Tests

VORTEX_ROOT ?= ../../../vortex

# Compiler settings
CXX := g++
CXXFLAGS := -Wall -Wextra -O2 -std=c++17

# Include paths
INCLUDES := -I../include
INCLUDES += -I$(VORTEX_ROOT)/runtime/include
INCLUDES += -I$(VORTEX_ROOT)/build/hw

# Library paths
LDFLAGS := -L../lib -L$(VORTEX_ROOT)/build/runtime
LIBS := -lhip_vortex_runtime -lvortex

# Tests
TESTS := test_vecadd_host test_native_kernel

.PHONY: all clean

all: $(TESTS)

test_vecadd_host: test_vecadd_host.cpp ../lib/libhip_vortex_runtime.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_native_kernel: test_native_kernel.cpp ../lib/libhip_vortex_runtime.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TESTS)

# Run test with native Vortex kernel
run-native: test_native_kernel
	@echo "Running with native Vortex vecadd kernel..."
	LD_LIBRARY_PATH=$(VORTEX_ROOT)/build/runtime:$$LD_LIBRARY_PATH ./test_native_kernel

.PHONY: run-native
