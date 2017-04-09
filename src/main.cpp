#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>

extern void filter_wrapper(const cv::Mat& input, cv::Mat& output);

int main()
{
	// Read input file (image)
	std::string imagePath = "data/image.png";
	cv::Mat input = cv::imread(imagePath,0);
	if(input.empty()) {
		std::cout<<"Could not load image. Check location and try again."<<std::endl;
		std::cin.get();
		return -1;
	}

	cv::Size resize_size;
	resize_size.width = 480;
	resize_size.height = 1200;
	cv::resize(input,input,resize_size);
	cv::Mat output_gpu(input.rows,input.cols,CV_8UC1);
	cv::Mat output_cpu(input.rows,input.cols,CV_8UC1);	

	clock_t gpu_s = clock();
	filter_wrapper(input,output_gpu);
	clock_t gpu_e = clock();
	double gpu_time = (double(gpu_e - gpu_s) * 1000)/CLOCKS_PER_SEC;
	std::cout << "GPU Accelerated Median Filter took " << gpu_time << " ms.\n";	
	
	clock_t cpu_s = clock();
	cv::medianBlur(input,output_cpu,9);
	clock_t cpu_e = clock();
	double cpu_time = (double(cpu_e - cpu_s) * 1000)/CLOCKS_PER_SEC;
	std::cout << "CPU Accelerated Median Filter took " << cpu_time << " ms.\n";	

	cv::imshow("Output Image - GPU",output_gpu);
	cv::imshow("Output Image - CPU",output_cpu);
	cv::waitKey();

	return 0;
}
