#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>
#include "helper_cuda.h"

__device__ void sort(float *x, int n_size) {
	for (int i = 0; i < n_size-1; i++) {
		int min_idx = i;
		for (int j = i + 1; j < n_size; j++) {
			if(x[j] < x[min_idx])
				min_idx = j;
		}
		float temp = x[min_idx];
		x[min_idx] = x[i];
		x[i] = temp;
	}	
}

__global__ void cvrgb_to_gray(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		
		//Location of gray pixel in output
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		const unsigned char blue	= input[color_tid];
		const unsigned char green	= input[color_tid + 1];
		const unsigned char red		= input[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void rgb2grayscale_gpu(const cv::Mat& input, cv::Mat& output)
{
	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;
	unsigned char *d_input, *d_output;
	
	cudaError_t cudaStatus;	
	
	//Allocate device memory
	cudaStatus = cudaMalloc<unsigned char>(&d_input,colorBytes);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaMalloc<unsigned char>(&d_output,grayBytes);
	checkCudaErrors(cudaStatus);	
	
	//Copy data from OpenCV input image to device memory
	cudaStatus = cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);	
	
	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	//Launch the color conversion kernel
	cvrgb_to_gray<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	//Synchronize to check for any kernel launch errors
	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);	

	//Copy back data from destination device meory to OpenCV output image
	cudaStatus = cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);	

	//Free the device memory
	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);	
}


