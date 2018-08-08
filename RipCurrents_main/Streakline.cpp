#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name

#include "Streakline.hpp"


typedef cv::Point_<float> Pixel2;

Streakline::Streakline() {
	numberOfVertices = 1;
}



