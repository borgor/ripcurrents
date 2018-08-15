#ifndef __CV_STREAKLINE_H
#define __CV_STREAKLINE_H

using namespace cv;

typedef cv::Point_<float> Pixel2;

class Streakline {
    public:
        int numberOfVertices;
        Pixel2 generationPoint;
        std::vector<Pixel2> vertices;
        int frameCount;

        Streakline(Pixel2 pixel);

        void drawLine();

        void runLK(UMat u_prev, UMat u_current, Mat& outImg);
};
#endif
