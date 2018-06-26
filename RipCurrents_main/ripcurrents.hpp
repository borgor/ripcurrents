#ifndef __RIPCURRENTS_HPP_INCLUDE__
#define __RIPCURRENTS_HPP_INCLUDE__

#define XDIM 640   // Dimensions to resize to
#define YDIM 480

#define HIST_BINS 50 // Number of bins for finding thresholds
#define HIST_DIRECTIONS 36 // Number of 2d histogram directions
#define HIST_RESOLUTION 20

#define BUFFER_FRAME 100 // Number of buffered frames

#define GRID_COUNT 15 // number of arrows per row and col

using namespace cv;

typedef cv::Point3_<uchar> Pixelc;
typedef cv::Point_<float> Pixel2;
typedef cv::Point3_<float> Pixel3;

void streamline_field(Pixel2 * pt, float* distancetraveled, int xoffset, int yoffset, cv::Mat flow, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]);
void streamline(Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]);
void display_histogram(int hist2d[HIST_DIRECTIONS][HIST_BINS],int histsum2d[HIST_DIRECTIONS]
					,float UPPER2d[HIST_DIRECTIONS], float UPPER, float prop_above_upper[HIST_DIRECTIONS]);
double timediff();

void streamline_displacement(Mat& streamfield, Mat& streamoverlay_color);
void streamline_total_motion(Mat& streamlines_distance, Mat& streamoverlay_color);
void streamline_ratio(Mat& streamfield, Mat& streamlines_distance, Mat& streamoverlay_color);

void streamline_positions(Mat& streamlines_mat, Mat& streamline_density);

void get_streamlines(Mat& streamout, Mat& streamoverlay_color, Mat& streamoverlay, int streamlines, Pixel2 streampt[], int framecount, int totalframes, Mat& current, float UPPER, float prop_above_upper[]);

void create_histogram(Mat current, int hist[HIST_BINS], int& histsum, int hist2d[HIST_DIRECTIONS][HIST_BINS]
	 				,int histsum2d[HIST_DIRECTIONS], float& UPPER, float UPPER2d[HIST_DIRECTIONS], float prop_above_upper[HIST_DIRECTIONS]);

void stabilizer(Mat current, Mat current_prev);

void globalOrientation(UMat u_f1, UMat u_f2, Mat& hist_gray);

void averageVector(std::vector<Mat> buffer, Mat& current, int update_ith_buffer, Mat& average, Mat& average_color, double** grid, float max_displacement, float UPPER);

void create_flow(Mat current, Mat waterclass, Mat accumulator2, float UPPER, float MID, float LOWER, float UPPER2d[HIST_DIRECTIONS]);

void create_accumulationbuffer(Mat& accumulator, Mat accumulator2, Mat& out, Mat outmask, int framecount);

void create_edges(Mat& outmask);

void create_output(Mat& subframe, Mat outmask);

void get_delta(Pixel2 * pt, int xoffset, int yoffset, cv::Mat flow, float dt, float UPPER);

#endif