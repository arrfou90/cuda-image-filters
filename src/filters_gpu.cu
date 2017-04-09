#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "helper_cuda.h"

const int BLOCKDIM = 32;
__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

__device__ void sort(float *x, int n_size) {
	// iterate over reference vector
	for (int i = 0; i < n_size-1; i++) {
		// initialize minimum element index
		int min_idx = i;
		// compare against rest of the elements
		for (int j = i + 1; j < n_size; j++) {
			// comparison
			if(x[j] < x[min_idx])
				min_idx = j;
		}
		// swap elements with minimum element
		float temp = x[min_idx];
		x[min_idx] = x[i];
		x[i] = temp;
	}	
}

__device__ int index(int x, int y, int width) 
{
	return (y * width) + x;
}

__global__ void median_filter_2d(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep, int kernel_size_r)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x<width) && (y<height))
	{
		const int color_tid = y * colorWidthStep + x;
		float xs[11*11];
		int xs_size = 0;

		for (int x_iter = x - kernel_size_r; x_iter <= x + kernel_size_r; x_iter ++) {
			for (int y_iter = y - kernel_size_r; y_iter <= y + kernel_size_r; y_iter++) {
				if (0<=x_iter && x_iter < colorWidthStep && 0 <= y_iter && y_iter < height) {
					xs[xs_size++] = input[y_iter * colorWidthStep + x_iter];
				}
			}
		}
		sort(xs,xs_size);
		output[color_tid] = static_cast<unsigned char>(xs[xs_size/2]);
	}
}

__global__ void median_filter_2d_sm(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep, int kernel_size_r)
{
	__shared__ int sharedPixels[BLOCKDIM + FILTER_SIZE][BLOCKDIM + FILTER_SIZE];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int xBlockLimit_max = blockDim.x - FILTER_HALFSIZE - 1;
	int yBlockLimit_max = blockDim.y - FILTER_HALFSIZE - 1;
	int xBlockLimit_min = FILTER_HALFSIZE;
	int yBlockLimit_min = FILTER_HALFSIZE;

	if (threadIdx.x > xBlockLimit_max && threadIdx.y > yBlockLimit_max) {
		int i = index(x + FILTER_HALFSIZE, y + FILTER_HALFSIZE, width);
	    	unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x > xBlockLimit_max && threadIdx.y < yBlockLimit_min) {
		int i = index(x + FILTER_HALFSIZE, y - FILTER_HALFSIZE, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min && threadIdx.y > yBlockLimit_max) {
		int i = index(x - FILTER_HALFSIZE, y + FILTER_HALFSIZE, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min && threadIdx.y < yBlockLimit_min) {
		int i = index(x - FILTER_HALFSIZE, y - FILTER_HALFSIZE, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y] = pixel;
	}
	if (threadIdx.x < xBlockLimit_min) {
		int i = index(x - FILTER_HALFSIZE, y, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x][threadIdx.y + FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.x > xBlockLimit_max) {
		int i = index(x + FILTER_HALFSIZE, y, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;
	}
	if (threadIdx.y < yBlockLimit_min) {
		int i = index(x, y - FILTER_HALFSIZE, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y] = pixel;
	}
	if (threadIdx.y > yBlockLimit_max) {
		int i = index(x, y + FILTER_HALFSIZE, width);
		unsigned int pixel = input[i];
		sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
	}
	int i = index(x, y, width);
	unsigned int pixel = input[i];
	sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;

	__syncthreads();

	if((x<width) && (y<height))
	{
		const int color_tid = y * colorWidthStep + x;
		float xs[11*11];
		int xs_size = 0;

		for (int x_iter = 0; x_iter < kernel_size_r; x_iter ++) {
			for (int y_iter = 0; y_iter < kernel_size_r; y_iter++) {
				if (0<=x_iter && x_iter < colorWidthStep && 0 <= y_iter && y_iter < height) {
					xs[xs_size++] = sharedPixels[threadIdx.x + x_iter][threadIdx.y + y_iter];
				}
			}
		}
		sort(xs,xs_size);
		output[color_tid] = static_cast<unsigned char>(xs[xs_size/2]);
	}
}

void filter_wrapper(const cv::Mat& input, cv::Mat& output)
{
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;
	unsigned char *d_input, *d_output;
	const int kernel = 5;

	//printf("ColorBytes = %d | grayBytes = %d\n",colorBytes,grayBytes);
	
	cudaError_t cudaStatus;	
	
	cudaStatus = cudaMalloc<unsigned char>(&d_input,colorBytes);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaMalloc<unsigned char>(&d_output,grayBytes);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);	
	
	const dim3 block(BLOCKDIM,BLOCKDIM);
	const dim3 grid(input.cols / BLOCKDIM, input.rows / BLOCKDIM);

	const int colwidstep = input.step;
	const int graywidstep = output.step;

	//printf("Color Width Step = %d | Gray Width Step = %d \n",colwidstep,graywidstep);

	median_filter_2d_sm<<<grid,block>>>(d_input,d_output,input.cols,input.rows,colwidstep,graywidstep,kernel);

	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);	
}


