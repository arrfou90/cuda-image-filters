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

extern void rgb2grayscale_gpu(const cv::Mat& input, cv::Mat& output);

int main()
{
	// Read input file (image)
	std::string imagePath = "data/image.png";
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);
	cv::Mat output(input.rows,input.cols,CV_8UC1);
	if(input.empty()) {
		std::cout<<"Could not load image. Check location and try again."<<std::endl;
		std::cin.get();
		return -1;
	}

	// Call Wrapper
	rgb2grayscale_gpu(input,output);

	cv::imshow("Input Image",input);
	cv::imshow("Output Image",output);
	cv::waitKey();

	return 0;
}