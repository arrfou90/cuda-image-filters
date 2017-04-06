CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all:
	nvcc -I. -arch=sm_30 -c src/filters_gpu.cu -o build/filters_gpu.o
	g++ -o build/filters_gpu src/main.cpp build/filters_gpu.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

clean: 
	@rm -rf *.o build/filters_gpu build/filters_gpu
	@rm -rf *~
