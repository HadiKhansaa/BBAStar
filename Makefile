# Public build targets for the CUDA bidirectional A* implementation.

BIN_DIR := bin
SRC_DIR := src
INC_DIR := include

TARGET := astar_bidirectional
TARGET_DEBUG := astar_bidirectional_debug

CUDA_ARCH ?= sm_89
NVCC ?= nvcc
HOST_COMPILER ?= cl

BIDIRECTIONAL_SOURCES := \
	$(SRC_DIR)/main_astar_bidirectional.cu \
	$(SRC_DIR)/bidirectional_astar.cu \
	$(SRC_DIR)/grid_generation.cu

NVCC_FLAGS := -I $(INC_DIR) -ccbin="$(HOST_COMPILER)" -arch=$(CUDA_ARCH)
RELEASE_FLAGS := -O3 -use_fast_math
DEBUG_FLAGS := -G -g -DDEBUG

TARGET_PATH := $(BIN_DIR)/$(TARGET).exe
TARGET_DEBUG_PATH := $(BIN_DIR)/$(TARGET_DEBUG).exe

.PHONY: all bidirectional bidirectional_debug debug clean help

all: bidirectional

help:
	@echo Supported targets:
	@echo   make                  Build the release CUDA binary.
	@echo   make bidirectional    Alias for the release build.
	@echo   make debug            Build the debug CUDA binary.
	@echo   make clean            Remove generated binaries and intermediate files.
	@echo
	@echo Optional variables:
	@echo   CUDA_ARCH=sm_89       Set the target GPU architecture passed to nvcc.
	@echo   HOST_COMPILER=cl      Override the MSVC host compiler command.

bidirectional: $(TARGET_PATH)

debug: bidirectional_debug

bidirectional_debug: $(TARGET_DEBUG_PATH)

$(BIN_DIR):
	mkdir $(BIN_DIR) 2>NUL || echo Bin directory already exists.

$(TARGET_PATH): $(BIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(RELEASE_FLAGS) -o $@ $^

$(TARGET_DEBUG_PATH): $(BIDIRECTIONAL_SOURCES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o $@ $^

clean:
	del /F /Q $(BIN_DIR)\$(TARGET).exe       2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\$(TARGET_DEBUG).exe 2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.pdb               2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.exp               2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.lib               2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.obj               2>NUL || echo Nothing to clean.
