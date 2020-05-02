EXECUTABLE := run_cuda

CU_FILES   := gat_cuda.cu

CU_DEPS    := gat_cuda.h

CC_FILES   := run_cuda.cpp utils_cuda.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -std=c++11 -m64
CXXFLAGS=-O3 -Wall -g
CUDA=cuda
LDFLAGS=-L/usr/local/depot/$(CUDA)/lib64/ -lcudart
NVCC=/usr/local/depot/$(CUDA)/bin/nvcc
NVCCFLAGS=-O3 -m64 -arch=compute_61 -g

OBJS=$(OBJDIR)/run_cuda.o  $(OBJDIR)/utils_cuda.o $(OBJDIR)/gat_cuda.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
