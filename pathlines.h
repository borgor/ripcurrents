#ifndef __CV_PATHLINES_H
#define __CV_PATHLINES_H

typedef cv::Point_<float> Pixel2;

int streamline (Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt);

#endif
