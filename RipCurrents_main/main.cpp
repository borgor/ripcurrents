#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <vector>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name
#include <opencv2/optflow/motempl.hpp>

#include "ripcurrents.hpp"
#include "Streakline.hpp"

int compute_streaklines(VideoCapture video);
int compute_streamlines(VideoCapture video);

int main(int argc, char** argv) {

	//Video I/O
	if (argc == 1){
		std::cout << " Input video not found" << std::endl;
		exit(1);
	}

	VideoCapture video;
	video = VideoCapture(argv[1]);
	if (!video.isOpened())
	{
		std::cout << "!!! Input video could not be opened" << std::endl;
		exit(1);
	}

	// compute_streaklines(video);
	compute_streamlines(video);
}

int compute_streaklines(VideoCapture video) {

	// set up output videos
	String video_name = "streaklines";
	VideoWriter video_output( video_name + ".mp4",CV_FOURCC('X','2','6','4'), 30, cv::Size(XDIM,YDIM),true);

	if (!video_output.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		return -1;
	}

	// OpenCV matrices to load images
	Mat frame;		// raw current frame image
	Mat resized_frame;	// resized current frame image
	Mat grayscaled_frame;			// gray scaled current frame image
	
	// OpenCL/GPU matrices
	UMat u_current;		// UMat current frame
	UMat u_prev;		// UMat previous frame
	

	// vector of streakliines
	std::vector<Streakline> streaklines;

	// initialize streaklines with seed generation locations
	# define MAX_STREAKLINES 5
	for (int s = 0; s < MAX_STREAKLINES; s++) {
		streaklines.push_back(Streakline(Pixel2(rand()%XDIM,rand()%YDIM)));
	}

	//Preload a frame
	video.read(frame);
	if(frame.empty()) exit(1);
	resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_AREA);
	cvtColor(resized_frame,grayscaled_frame,COLOR_BGR2GRAY);
	grayscaled_frame.copyTo(u_prev);


	// read and process every frame
	for( int framecount = 1; true; framecount++){

		// read current frame
		video.read(frame);
		printf("Frames read: %d\n",framecount);
		
		if(frame.empty()) break;
		

		// prepare input image
		resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(resized_frame,grayscaled_frame,COLOR_BGR2GRAY);
		grayscaled_frame.copyTo(u_current);

		Mat features;	// output image
		resized_frame.copyTo(features);

		// run calcOpticalFlowPyrLK on each streakline
		for ( int i = 0; i < (int)streaklines.size(); i++ ) {
			streaklines[i].runLK(u_prev, u_current, features);
		}
		imshow("streaklines", features);
		video_output.write(features);
		
		// prepare for next frame
		u_current.copyTo(u_prev);

		// end with Esc key on any window
		int c = waitKey(1);
		if ( c == 27) break;

		// stop and restart with any key
		if ( c != -1 && c != 27 ) {
			waitKey(0);
		}

	}

	// clean up
	video_output.release();
	destroyAllWindows();

	return 1;
}

int compute_streamlines(VideoCapture video) {

	// set up output videos
	String video_name = "streamlines";
	VideoWriter video_output( video_name + ".mp4",CV_FOURCC('X','2','6','4'), 30, cv::Size(XDIM,YDIM),true);

	if (!video_output.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		return -1;
	}

	// OpenCV matrices to load images
	Mat frame;			// raw current frame image
	Mat resized_frame;	// resized current frame image
	Mat grayscaled_frame;			// gray scaled current frame image
	Mat current = Mat::zeros(YDIM,XDIM,CV_32FC2);;		// Mat output velocity field of OpticalFlow
	
	// OpenCL/GPU matrices
	UMat u_current;		// UMat current frame
	UMat u_prev;		// UMat previous frame
	UMat u_flow;		// output velocity field of OpticalFlow

	// streamlines matrices
	Mat streamoverlay = Mat::zeros(Size(XDIM, YDIM), CV_8UC1);
	Mat streamoverlay_color = Mat::zeros(Size(XDIM, YDIM), CV_8UC3);
	int totalframes = (int) video.get(CAP_PROP_FRAME_COUNT);
	

	// Histogram
	// Some thresholds to mask out any remaining jitter, and strong waves. Don't know how to calculate them at runtime, so they're arbitrary.
	float LOWER =  0.2;
	float MID  = .5;
	int hist[HIST_BINS] = {0};	//histogram
	int histsum = 0;
	float UPPER = 45.0;		// UPPER can be determined programmatically
	int hist2d[HIST_DIRECTIONS][HIST_BINS] = {{0}};
	int histsum2d[HIST_DIRECTIONS] = {0};
	float UPPER2d[HIST_DIRECTIONS] = {0};

	float prop_above_upper[HIST_DIRECTIONS] = {0};


	// Preload a frame
	video.read(frame);
	if(frame.empty()) exit(1);
	resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_AREA);
	cvtColor(resized_frame,grayscaled_frame,COLOR_BGR2GRAY);
	grayscaled_frame.copyTo(u_prev);

	


	// discrete streamline seed points
	# define MAX_STREAMLINES 20
	Pixel2 streampt[MAX_STREAMLINES];
	int streamlines = MAX_STREAMLINES/2;

	sranddev();
	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}


	namedWindow("streamlines", WINDOW_AUTOSIZE );

	int framecount = 0;
	// read and process every frame
	for( framecount = 1; true; framecount++){

		// read current frame
		video.read(frame);
		printf("Frames read : %d\n",framecount);
		
		if(frame.empty()) break;
		

		// prepare input image
		resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(resized_frame,grayscaled_frame,COLOR_BGR2GRAY);
		grayscaled_frame.copyTo(u_current);


		// Run optical flow farneback
		//calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		//current = u_flow.getMat(ACCESS_READ);


		// circle sample input vector field
		for ( int row = 0; row < YDIM; row++ ){
			for ( int col = 0; col < XDIM; col++ ){
				current.at<Pixel2>(row,col).x = -(row - YDIM / 2.0) / YDIM;
				current.at<Pixel2>(row,col).y = (col - XDIM / 2.0) / XDIM;
			}
		}
		


		// draw streamlines
		Mat streamout;
		resized_frame.copyTo(streamout);
		get_streamlines(streamout, streamoverlay_color, streamoverlay, streamlines, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		imshow("streamlines",streamout);
		video_output.write(streamout);
		
		
		// prepare for next frame
		u_current.copyTo(u_prev);

		// Construct histograms to get thresholds
		// Figure out what "slow" or "fast" is
		
		//create_histogram(current,  hist, histsum, hist2d, histsum2d, UPPER, UPPER2d, prop_above_upper);

		// end with Esc key on any window
		int c = waitKey(1);
		if ( c == 27) break;

		// stop and restart with any key
		if ( c != -1 && c != 27 ) {
			waitKey(0);
		}

	}

	// clean up
	video_output.release();
	destroyAllWindows();

	return 0;
}