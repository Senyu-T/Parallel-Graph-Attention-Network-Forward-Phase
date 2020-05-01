all: run_cuda

run_cuda: gat_cuda.o
	g++ -std=c++11 -o run_cuda run_cuda.cpp utils_cuda.cpp -L/usr/local/depot/cuda/lib64/ -lcuda -lcudart gat_cuda.o -lm

gat_cuda.o:
	nvcc -c -arch=sm_30 gat_cuda.cu

clean:
	rm -f *.o run_cuda
