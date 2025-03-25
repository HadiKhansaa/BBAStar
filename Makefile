# Makefile for building A* project with CUDA + MSVC on Windows

# Project configuration
TARGET        := astar
BIDIRECTIONAL_TARGET := astar_bidirectional
DEBUG_TARGET  := astar_debug
BIN_DIR       := bin
SRC_DIR       := src
INC_DIR       := include
CU_FILES      := $(SRC_DIR)/main_astar.cu
CPP_FILES     := $(SRC_DIR)/grid_generation.cpp

# Compiler settings
NVCC          := nvcc
HOST_COMPILER := cl # Adjust if needed (e.g. full path)

# Common NVCC flags (adjust as necessary)
NVCC_FLAGS    := -I $(INC_DIR) -ccbin="$(HOST_COMPILER)" -arch=native

# Default rule: build release target
all: $(BIN_DIR)/$(TARGET)

# Release build rule
$(BIN_DIR)/$(TARGET): $(CU_FILES) $(CPP_FILES)
	mkdir $(BIN_DIR) 2>NUL || echo Bin directory already exists.
	$(NVCC) $(NVCC_FLAGS) -O3 -o $@ $^

# Debug build rule
debug: $(BIN_DIR)/$(DEBUG_TARGET)

$(BIN_DIR)/$(DEBUG_TARGET): $(CU_FILES) $(CPP_FILES)
	mkdir $(BIN_DIR) 2>NUL || echo Bin directory already exists.
	$(NVCC) $(NVCC_FLAGS) -G -g -DDEBUG -o $@ $^

# Bidirectional build rule
bidirectional:
	$(MAKE) TARGET=astar_bidirectional CU_FILES=$(SRC_DIR)/main_astar_bidirectional.cu all


$(BIN_DIR)/astar_bidirectional_debug: $(SRC_DIR)/main_astar_bidirectional.cu $(CPP_FILES)
	mkdir $(BIN_DIR) 2>NUL || echo Bin directory already exists.
	$(NVCC) $(NVCC_FLAGS) -G -g -DDEBUG -o $@ $^

bidirectional_debug: $(BIN_DIR)/astar_bidirectional_debug

# Clean build outputs
clean:
	del /F /Q $(BIN_DIR)\$(TARGET).exe        2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\$(DEBUG_TARGET).exe   2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\$(BIDIRECTIONAL_TARGET).exe   2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\$(BIDIRECTIONAL_TARGET)_debug.exe   2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.pdb                 2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.exp                 2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.lib                 2>NUL || echo Nothing to clean.
	del /F /Q $(BIN_DIR)\*.obj                 2>NUL || echo Nothing to clean.
