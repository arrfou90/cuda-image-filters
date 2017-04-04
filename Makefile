CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all:
	nvcc -I. -arch=sm_30 -c src/rgb2gray_gpu.cu -o build/rgb2gray_gpu.o
	g++ -o build/rgb2gray src/main.cpp build/rgb2gray_gpu.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

clean: 
	@rm -rf *.o build/rgb2gray build/rgb2gray_gpu
	@rm -rf *~
