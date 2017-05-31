//
// Created by Boris Gorshenev on 4/20/17.
//
#include <opencv2/opencv.hpp>


#define XDIM 640.0   //Dimensions to resize to
#define YDIM 480.0

#ifndef OPENCV_TEST_RIPCURRENTS_H_H
#define OPENCV_TEST_RIPCURRENTS_H_H
void wheel();
int rip_main(cv::VideoCapture video, cv::VideoWriter video_out);
#endif //OPENCV_TEST_RIPCURRENTS_H_H
