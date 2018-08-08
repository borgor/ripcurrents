#ifndef __CV_STREAKLINE_H
#define __CV_STREAKLINE_H

using namespace cv;
using namespace std;

typedef cv::Point_<float> Pixel2;

class Streakline {
    public:
        int numberOfVertices;
        Streakline(void);
        void drawLine();
};
#endif
