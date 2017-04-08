#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "helper_cuda.h"

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

__global__ void median_filter_2d(float *image_in, float *image_out,
				int size, int dim_2, int dim_3, 
				int kernel_size_r)
{
	// find thread id in global memory organization
	int thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
	// if within image limits (max size)
	if (thread_id < size) {
		// find x and y indices
		int x = thread_id % dim_3; // dim3 is the size of the row
		int y = thread_id / dim_3; // equivalently #cols * size
		float xs[11*11]; // allocate some memory for presort
		int xs_size = 0;
		// iterate over image x axis
		for (int x_iter = x - kernel_size_r; x_iter <= x + kernel_size_r; x_iter ++) {
			// iterate over image y axis
			for (int y_iter = y - kernel_size_r; y_iter <= y + kernel_size_r; y_iter++) {
				// stay within image block dimensions
				if (0<=x_iter && x_iter < dim_3 && 0 <= y_iter && y_iter < dim_2) {
					// fill up pre-sorted vector
					// image_in[row_offset*row_size + col_offset]
					xs[xs_size++] = image_in[y_iter * dim_3 + x_iter];
				}
			}
		}
		// sort the given vector using the device method sort(*x,n)
		sort(xs,xs_size);
		// allocate the median of the sorted vector to the pixel at image out
		image_out[thread_id] = xs[xs_size/2];
	}
}


__global__ void cvrgb_to_gray(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		const int color_tid = yIndex * colorWidthStep + xIndex;
		const int gray_tid  = yIndex * grayWidthStep + xIndex;
		const float gray = input[color_tid];
		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void filter_wrapper(const cv::Mat& input, cv::Mat& output)
{
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;
	unsigned char *d_input, *d_output;

	printf("ColorBytes = %d | grayBytes = %d\n",colorBytes,grayBytes);
	
	cudaError_t cudaStatus;	
	
	cudaStatus = cudaMalloc<unsigned char>(&d_input,colorBytes);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaMalloc<unsigned char>(&d_output,grayBytes);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);	
	
	const dim3 block(16,16);

	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	const int colwidstep = input.step;
	const int graywidstep = output.step;

	printf("Color Width Step = %d | Gray Width Step = %d \n",colwidstep,graywidstep);
	cvrgb_to_gray<<<grid,block>>>(d_input,d_output,input.cols,input.rows,colwidstep,graywidstep);

	//median_filter_2d<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	cudaStatus = cudaDeviceSynchronize();
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);	

	cudaStatus = cudaFree(d_input);
	checkCudaErrors(cudaStatus);	
	cudaStatus = cudaFree(d_output);
	checkCudaErrors(cudaStatus);	
}


