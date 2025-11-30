# Makefile for hip_vortex_runtime
# Minimal HIP runtime library that maps to Vortex API

# Paths
VORTEX_ROOT ?= ../../vortex
VORTEX_BUILD := $(VORTEX_ROOT)/build/runtime
VORTEX_LIB := $(VORTEX_BUILD)/libvortex.so

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

# Compiler settings
CXX := g++
AR := ar
CXXFLAGS := -Wall -Wextra -O2 -fPIC
CXXFLAGS += -I$(INC_DIR)
CXXFLAGS += -I$(VORTEX_ROOT)/runtime/include
CXXFLAGS += -I$(VORTEX_ROOT)/build/hw  # For VX_config.h

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Output library
LIB := lib/libhip_vortex_runtime.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS) | lib
	@echo "Creating static library: $@"
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

lib:
	mkdir -p lib

clean:
	rm -rf $(BUILD_DIR) lib

install: $(LIB)
	@echo "Installing hip_vortex_runtime..."
	mkdir -p ../../lib
	cp $(LIB) ../../lib/
	mkdir -p ../../include/hip
	cp $(INC_DIR)/hip_vortex_runtime.h ../../include/hip/

.PHONY: test
test:
	@echo "Runtime library sources:"
	@echo "  $(SRCS)"
	@echo "Object files:"
	@echo "  $(OBJS)"
