# Public build targets for the CUDA bidirectional A* implementation.

BIN_DIR := bin
SRC_DIR := src
INC_DIR := Include

TARGET := astar_bidirectional
TARGET_DEBUG := astar_bidirectional_debug
UNIDIRECTIONAL_TARGET := astar_unidirectional
UNIDIRECTIONAL_TARGET_DEBUG := astar_unidirectional_debug

CUDA_ARCH ?= sm_89
NVCC ?= nvcc
HOST_COMPILER ?= g++

BIDIRECTIONAL_SOURCES := \
	$(SRC_DIR)/main_astar_bidirectional.cu \
	$(SRC_DIR)/bidirectional_astar.cu \
	$(SRC_DIR)/grid_generation.cu

UNIDIRECTIONAL_SOURCES := \
	$(SRC_DIR)/main_astar_unidirectional.cu \
	$(SRC_DIR)/unidirectional_astar.cu \
	$(SRC_DIR)/grid_generation.cu

NVCC_FLAGS := -I $(INC_DIR) -ccbin=$(HOST_COMPILER) -arch=$(CUDA_ARCH)
RELEASE_FLAGS := -O3 -use_fast_math
DEBUG_FLAGS := -G -g -DDEBUG

TARGET_PATH := $(BIN_DIR)/$(TARGET)
TARGET_DEBUG_PATH := $(BIN_DIR)/$(TARGET_DEBUG)
UNIDIRECTIONAL_TARGET_PATH := $(BIN_DIR)/$(UNIDIRECTIONAL_TARGET)
UNIDIRECTIONAL_TARGET_DEBUG_PATH := $(BIN_DIR)/$(UNIDIRECTIONAL_TARGET_DEBUG)

.PHONY: all bidirectional bidirectional_debug unidirectional unidirectional_debug debug clean help

all: bidirectional

help:
	@echo Supported targets:
	@echo   make                  Build the release CUDA binary.
	@echo   make bidirectional    Alias for the release build.
	@echo   make unidirectional   Build the unidirectional CUDA binary.
	@echo   make debug            Build the debug CUDA binary.
	@echo   make unidirectional_debug  Build the debug unidirectional CUDA binary.
	@echo   make clean            Remove generated binaries and intermediate files.
	@echo
	@echo Optional variables:
	@echo   CUDA_ARCH=sm_89       Set the target GPU architecture passed to nvcc.
	@echo   HOST_COMPILER=g++     Override the GCC-compatible host compiler command.

bidirectional: $(TARGET_PATH)

unidirectional: $(UNIDIRECTIONAL_TARGET_PATH)

debug: bidirectional_debug

bidirectional_debug: $(TARGET_DEBUG_PATH)

unidirectional_debug: $(UNIDIRECTIONAL_TARGET_DEBUG_PATH)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET_PATH): $(BIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(RELEASE_FLAGS) -o $@ $^

$(TARGET_DEBUG_PATH): $(BIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o $@ $^

$(UNIDIRECTIONAL_TARGET_PATH): $(UNIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(RELEASE_FLAGS) -o $@ $^

$(UNIDIRECTIONAL_TARGET_DEBUG_PATH): $(UNIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET_PATH) \
		$(TARGET_DEBUG_PATH) \
		$(UNIDIRECTIONAL_TARGET_PATH) \
		$(UNIDIRECTIONAL_TARGET_DEBUG_PATH) \
		$(BIN_DIR)/*.pdb \
		$(BIN_DIR)/*.exp \
		$(BIN_DIR)/*.lib \
		$(BIN_DIR)/*.obj
