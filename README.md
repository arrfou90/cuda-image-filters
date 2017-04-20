# Cuda Image Filters on NVIDIA TX-2 & NVIDIA GTX1070
Implementation of GPU accelerated Median Filter and Bilateral Filter for Stereo Vision using CUDA and OpenCV for CIS601 - Special Topics in Computer Architecture : GPGPU Programming

To use this repository, make sure you have the following components installed:
* OpenCV 2.4
* CUDA 8.0

Clone the repository;

Once inside the repository launch the Makefile with the "make" command;

This repository already contains some test data in the "/data" folder; If you wish to use your own images, add a link to your desired image in *main.cpp* located in the "/src" folder;

Performance is measured over 10 attempts, and the first warm-up kernel launch is omitted in each case. You can change this by modifying the **attempts** variable in *main.cpp*;

# Results

## Median Filter

Original Image:

![Original Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/imagemedian.png)

CUDA Implementation:

![CUDA Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/gpu_median_result.png)

OpenCV Implementation:

![OpenCV Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/cpu_median_result.png)


## Bilateral Filter

Original Image:

![Original Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/imagebilateral.png)

CUDA Implementation:

![CUDA Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/gpu_bilateral_result.png)

OpenCV Implementation:

![OpenCV Image](https://github.com/ShreyasSkandan/cuda-image-filters/blob/master/data/cpu_bilateral_result.png)
