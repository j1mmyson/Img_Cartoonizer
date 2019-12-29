
#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int main() {
	
	Mat src = imread("C:/Users/bwson/source/son.jpg", IMREAD_COLOR);
	Mat grad;
	int scale = 1;
	int delta = 0;
	
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat dst;
	Mat result;
	Mat hsv;
	Mat colormap(Size(5, 5), CV_8UC3);

	//Red
	colormap.at<Vec3b>(0, 0)[0] = 0; 
	colormap.at<Vec3b>(0, 0)[1] = 0; 
	colormap.at<Vec3b>(0, 0)[2] = 255; 

	//Orange
	colormap.at<Vec3b>(0, 1)[0] = 0; 
	colormap.at<Vec3b>(0, 1)[1] = 102; 
	colormap.at<Vec3b>(0, 1)[2] = 255; 

	//Yello
	colormap.at<Vec3b>(0, 2)[0] = 0; 
	colormap.at<Vec3b>(0, 2)[1] = 255; 
	colormap.at<Vec3b>(0, 2)[2] = 255; 

	//Skin
	colormap.at<Vec3b>(0, 3)[0] = 153;
	colormap.at<Vec3b>(0, 3)[1] = 204;
	colormap.at<Vec3b>(0, 3)[2] = 255;

	//Skin2
	colormap.at<Vec3b>(0, 4)[0] = 204;
	colormap.at<Vec3b>(0, 4)[1] = 204;
	colormap.at<Vec3b>(0, 4)[2] = 255;

	//Green
	colormap.at<Vec3b>(1, 0)[0] = 0; 
	colormap.at<Vec3b>(1, 0)[1] = 255; 
	colormap.at<Vec3b>(1, 0)[2] = 0; 

	//Blue
	colormap.at<Vec3b>(1, 1)[0] = 255; 
	colormap.at<Vec3b>(1, 1)[1] = 0; 
	colormap.at<Vec3b>(1, 1)[2] = 0; 

	//Indigo
	colormap.at<Vec3b>(1, 2)[0] = 102; 
	colormap.at<Vec3b>(1, 2)[1] = 0; 
	colormap.at<Vec3b>(1, 2)[2] = 51; 

	//Brown
	colormap.at<Vec3b>(1, 3)[0] = 51;
	colormap.at<Vec3b>(1, 3)[1] = 102;
	colormap.at<Vec3b>(1, 3)[2] = 51;

	//Brown2
	colormap.at<Vec3b>(1, 4)[0] = 34;
	colormap.at<Vec3b>(1, 4)[1] = 34;
	colormap.at<Vec3b>(1, 4)[2] = 178;

	//MidGreen
	colormap.at<Vec3b>(2, 0)[0] = 0; 
	colormap.at<Vec3b>(2, 0)[1] = 102; 
	colormap.at<Vec3b>(2, 0)[2] = 0; 

	//Black
	colormap.at<Vec3b>(2, 1)[0] = 0; 
	colormap.at<Vec3b>(2, 1)[1] = 0; 
	colormap.at<Vec3b>(2, 1)[2] = 0; 

	//White
	colormap.at<Vec3b>(2, 2)[0] = 255; 
	colormap.at<Vec3b>(2, 2)[1] = 255; 
	colormap.at<Vec3b>(2, 2)[2] = 255; 

	//
	colormap.at<Vec3b>(2, 3)[0] = 0;
	colormap.at<Vec3b>(2, 3)[1] = 0;
	colormap.at<Vec3b>(2, 3)[2] = 0;

	//karky
	colormap.at<Vec3b>(2, 3)[0] = 140;
	colormap.at<Vec3b>(2, 3)[1] = 230;
	colormap.at<Vec3b>(2, 3)[2] = 240;

	//SkyBlue
	colormap.at<Vec3b>(3, 0)[0] = 255;
	colormap.at<Vec3b>(3, 0)[1] = 204;
	colormap.at<Vec3b>(3, 0)[2] = 102;

	//SkyGreen
	colormap.at<Vec3b>(3, 1)[0] = 102;
	colormap.at<Vec3b>(3, 1)[1] = 255;
	colormap.at<Vec3b>(3, 1)[2] = 102;

	//Gray
	colormap.at<Vec3b>(3, 2)[0] = 180;
	colormap.at<Vec3b>(3, 2)[1] = 180;
	colormap.at<Vec3b>(3, 2)[2] = 180;

	//SkyPurple
	colormap.at<Vec3b>(3, 3)[0] = 200;
	colormap.at<Vec3b>(3, 3)[1] = 200;
	colormap.at<Vec3b>(3, 3)[2] = 200;

	//Gray
	colormap.at<Vec3b>(3, 3)[0] = 204;
	colormap.at<Vec3b>(3, 3)[1] = 204;
	colormap.at<Vec3b>(3, 3)[2] = 204;


	//
	colormap.at<Vec3b>(4, 0)[0] = 102;
	colormap.at<Vec3b>(4, 0)[1] = 204;
	colormap.at<Vec3b>(4, 0)[2] = 102;
	
	//Skin3
	colormap.at<Vec3b>(4, 1)[0] = 102;
	colormap.at<Vec3b>(4, 1)[1] = 204;
	colormap.at<Vec3b>(4, 1)[2] = 255;

	//Pink??
	colormap.at<Vec3b>(4, 2)[0] = 215;
	colormap.at<Vec3b>(4, 2)[1] = 230;
	colormap.at<Vec3b>(4, 2)[2] = 250;

	//light yellow
	colormap.at<Vec3b>(4, 3)[0] = 224;
	colormap.at<Vec3b>(4, 3)[1] = 255;
	colormap.at<Vec3b>(4, 3)[2] = 255;

	//
	colormap.at<Vec3b>(4, 4)[0] = 63;
	colormap.at<Vec3b>(4, 4)[1] = 133;
	colormap.at<Vec3b>(4, 4)[2] = 205;

	
	Sobel(src, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(src, grad_y, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow("Image", src);
	imshow("Sobel", grad);

	Sobel(grad, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(grad, grad_y, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	imshow("2sobel", grad);

	dst = src - grad;

	imshow("dst", dst);

	GaussianBlur(dst, result, Size(3, 3), 0, 0);
	imshow("G B", result);

	for (int i = 0; i < result.rows; i++) {
		for (int j = 0; j < result.cols; j++) {
			int min = 99999;
			float sum = 0;
			for (int a = 0; a < 5; a++) {
				for (int b = 0; b < 5; b++) {
					sum = sum + pow(result.at<Vec3b>(i, j)[0] - colormap.at<Vec3b>(a, b)[0], 4.0); 
					sum = sum + pow(result.at<Vec3b>(i, j)[1] - colormap.at<Vec3b>(a, b)[1], 4.0);
					sum = sum + pow(result.at<Vec3b>(i, j)[2] - colormap.at<Vec3b>(a, b)[2], 4.0);
					if (sum < min) {
						min = sum;
					}
					sum = 0;
				}
			}
			for (int a = 0; a < 5; a++) {
				for (int b = 0; b < 5; b++) {
					sum = sum + pow(result.at<Vec3b>(i, j)[0] - colormap.at<Vec3b>(a, b)[0], 4.0);
					sum = sum + pow(result.at<Vec3b>(i, j)[1] - colormap.at<Vec3b>(a, b)[1], 4.0);
					sum = sum + pow(result.at<Vec3b>(i, j)[2] - colormap.at<Vec3b>(a, b)[2], 4.0);
					if (sum == min) {
						result.at<Vec3b>(i, j)[0] = colormap.at<Vec3b>(a, b)[0];
						result.at<Vec3b>(i, j)[1] = colormap.at<Vec3b>(a, b)[1];
						result.at<Vec3b>(i, j)[2] = colormap.at<Vec3b>(a, b)[2];
					}
					sum = 0;
				}
			}
		}
	}
	imshow("Fin", result);

		
	waitKey(0);
	return 0;



}
