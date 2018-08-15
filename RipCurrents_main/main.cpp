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

	compute_streaklines(video);
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