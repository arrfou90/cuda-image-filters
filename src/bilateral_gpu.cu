#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "helper_cuda.h"

const int BLOCKDIM = 32;
const int sigma1 = 50;
const int sigma2 = 50;

__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;


__device__ float exp(int i) { return exp((float) i); }

__global__ void bilateral_filter_2d(unsigned char* input, unsigned char* output, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x<width) && (y<height))
	{
		float sum = 0;
		float cnt = 0;
		const int color_tid = y * width + x;
		for (int i = -FILTER_HALFSIZE; i <= FILTER_HALFSIZE; i++) 
		{
			for (int j = -FILTER_HALFSIZE; j <= FILTER_HALFSIZE; j++) 
			{
				int yy = y + i;
				int xx = x + j;
				if (0 <= xx && xx < width && 0 <= yy && yy < height) 
				{
					float color_diff = input[yy * width + xx] - input[y * width + x];
					float v1 = exp(-(i * i + j * j) / (2 * sigma1 * sigma1));
					float v2 = exp(-(color_diff * color_diff) / (2 * sigma2 * sigma2));
					sum += input[yy * width + xx] * v1 * v2;
					cnt += v1 * v2;
				}
			}
		}
		output[color_tid] = sum / cnt;
	}
}


void bilateral_filter_wrapper(const cv::Mat& input, cv::Mat& output)
{
	unsigned char *d_input, *d_output;
	cudaError_t cudaStatus;	
	
	cudaStatus = cudaMalloc<unsigned char>(&d_input,input.rows*input.cols);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaMalloc<unsigned char>(&d_output,output.rows*output.cols);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(d_input,input.ptr(),input.rows*input.cols,cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);	
	
	const dim3 block(BLOCKDIM,BLOCKDIM);
	const dim3 grid(input.cols/BLOCKDIM, input.rows/BLOCKDIM);

	bilateral_filter_2d<<<grid,block>>>(d_input,d_output,input.cols,input.rows);

	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaMemcpy(output.ptr(),d_output,output.rows*output.cols,cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);	
}
