NVCC = nvcc
CXX = cl
CXXFLAGS = /O2
NVCCFLAGS = -O3

TARGET = astar.exe
SRCS = main_astar.cu .\CPU\grid_generation.cpp
OBJS = $(SRCS:.cu=.obj)
OBJS = $(OBJS:.cpp=.obj)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ 

%.obj: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.obj: %.cpp
	$(CXX) $(CXXFLAGS) /c $< /Fo$@

clean:
	del /Q $(OBJS) $(TARGET)

.PHONY: all clean