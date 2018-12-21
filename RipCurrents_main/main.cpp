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
int validate_streamlines(VideoCapture video);
int compute_timelines(VideoCapture video);
int compute_subtructAverageVector(VideoCapture video);
int timelinesOnSubtractAverageVector(VideoCapture video);
int compute_populationMap(VideoCapture video);
int compute_timelinesFarne(VideoCapture video);
int compute_subtructAverageVectorWithWindow(VideoCapture video);

String outputFileName = "default";

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

	if (argc == 3){
		outputFileName = argv[2];
	}

	// compute_streaklines(video);
	// compute_streamlines(video);
	// compute_timelines(video);
	// compute_subtructAverageVector(video);
	// compute_populationMap(video);
	// compute_timelinesFarne(video);
	 compute_subtructAverageVectorWithWindow(video);

	return 0;
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

	/*sranddev();
	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}*/
	// original seed point
	streamlines = 1;
	streampt[0] = Pixel2(300,300);


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
		calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		current = u_flow.getMat(ACCESS_READ);


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


int validate_streamlines(VideoCapture video) {

	// set up output videos
	String video_name = "streamlines_validate";
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
	Mat current = Mat::zeros(YDIM,XDIM,CV_32FC2);		// Mat output velocity field of OpticalFlow
	
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

	namedWindow("streamlines", WINDOW_AUTOSIZE );

	// read current frame
	video.read(frame);
	
	if(frame.empty()) return 1;
	

	// prepare input image
	resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_LINEAR);
	cvtColor(resized_frame,grayscaled_frame,COLOR_BGR2GRAY);
	grayscaled_frame.copyTo(u_current);


	// Run optical flow farneback
	//calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
	//current = u_flow.getMat(ACCESS_READ);


	Mat flow = Mat::zeros(YDIM,XDIM,CV_32FC2);;

	// circle sample input vector field
	for ( int row = 0; row < YDIM; row++ ){
		for ( int col = 0; col < XDIM; col++ ){
			flow.at<Pixel2>(row,col).x = -(row - YDIM / 2.0) / YDIM * 100;
			flow.at<Pixel2>(row,col).y = (col - XDIM / 2.0) / XDIM * 100;
		}
	}

	printf("%f\n", flow.at<Pixel2>(200,200).x);
	printf("%f\n", flow.at<Pixel2>(200,200).y);
	
	// draw streamlines
	Mat streamout;
	resized_frame.copyTo(streamout);

	// original seed point
	Pixel2 streampt = Pixel2(200,200);

	double dt = 0.03;
	int iteration = 3500;

	for( int i = 0; i < iteration; i++){

		if(i % 10 == 0 ) printf("%d iterations\n", i);
		
		float x = streampt.x;
		float y = streampt.y;
		
		int xind = (int) floor(x);
		int yind = (int) floor(y);
		float xrem = x - xind;
		float yrem = y - yind;
		
		if(xind < 1 || yind < 1 || xind + 2 > flow.cols || yind  + 2 > flow.rows)  //Verify array bounds
		{
			continue;
		}
		
		//Bilinear interpolation
		Pixel2 delta =		(*flow.ptr<Pixel2>(yind,xind))		* (1-xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind,xind+1))	* (xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind))	* (1-xrem)*(yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind+1))	* (xrem)*(yrem) ;
		
		
		
		Pixel2 newpt = streampt + delta * dt;

		//printf("%f\n", delta.x);
		//printf("%f\n", delta.y);
		
		cv::line(streamout, streampt, newpt, CV_RGB(100,0,0), 1, 8, 0);

		imshow("streamlines",streamout);
		video_output.write(streamout);
		
		streampt = newpt;

		// end with Esc key on any window
		int c = waitKey(1);
		if ( c == 27) break;
	}


	// clean up
	video_output.release();
	destroyAllWindows();

	return 0;
}


int compute_timelines(VideoCapture video) {

	// @ params
	Pixel2 lineStart = Pixel2(100,180);
	Pixel2 lineEnd = Pixel2(500,250);
	int numberOfVertices = 20;

	// set up output videos
	String video_name = "timelines";
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

	// initialize Timeline
	Timeline timeline = Timeline(lineStart, lineEnd, numberOfVertices);

	//Preload the first frame as previous frame
	video.read(frame);
	if(frame.empty()) exit(1);
	resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_LINEAR);
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

		Mat outImg;	// output image
		resized_frame.copyTo(outImg);

		timeline.runLK(u_prev, u_current, outImg);

		imshow("timelines", outImg);
		video_output.write(outImg);
		
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

int compute_subtructAverageVector(VideoCapture video) {

	// set up output videos
	String video_name = "subtructAverageVector";
	VideoWriter video_output( outputFileName + ".mp4",CV_FOURCC('X','2','6','4'), 30, cv::Size(XDIM,YDIM),true);

	if (!video_output.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		return -1;
	}

	// OpenCV matrices to load images
	Mat frame;			// raw current frame image
	Mat resized_frame;	// resized current frame image
	Mat grayscaled_frame;			// gray scaled current frame image
	Mat current;		// Mat output velocity field of OpticalFlow
	
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
	# define MAX_STREAMLINES 40
	Pixel2 streampt[MAX_STREAMLINES];
	int streamlines = MAX_STREAMLINES/2;

	sranddev();
	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}
	// original seed point
	// streamlines = 1;
	// streampt[0] = Pixel2(300,300);


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
		// 0.5, 2, 3, 2, 15, 1.2
		calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 20, 3, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		current = u_flow.getMat(ACCESS_READ);


		// draw streamlines
		Mat outImg;
		resized_frame.copyTo(outImg);

		subtructAverage(current);
		

		// draw streamlines
		Mat streamout;
		resized_frame.copyTo(streamout);
		// get_streamlines(streamout, streamoverlay_color, streamoverlay, streamlines, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		// imshow("streamlines",streamout);
		// video_output.write(streamout);


		
		vectorToColor(current, outImg);

		drawFrameCount(outImg, framecount);
		
		imshow("subtruct average vector",outImg);
		video_output.write(outImg);
		
		
		
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
	current.release();

	return 0;
}

int timelinesOnSubtractAverageVector(VideoCapture video) {

	// set up output videos
	String video_name = "subtructAverageVector";
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
	Mat current;		// Mat output velocity field of OpticalFlow
	
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
	# define MAX_STREAMLINES 40
	Pixel2 streampt[MAX_STREAMLINES];
	int streamlines = MAX_STREAMLINES/2;

	sranddev();
	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}
	// original seed point
	// streamlines = 1;
	// streampt[0] = Pixel2(300,300);


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
		calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		current = u_flow.getMat(ACCESS_READ);


		// draw streamlines
		Mat outImg;
		resized_frame.copyTo(outImg);

		subtructAverage(current);

		// draw streamlines
		Mat streamout;
		resized_frame.copyTo(streamout);
		get_streamlines(streamout, streamoverlay_color, streamoverlay, streamlines, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		imshow("streamlines",streamout);
		video_output.write(streamout);


		/*
		vectorToColor(current, outImg);
		
		imshow("subtruct average vector",outImg);
		video_output.write(outImg);
		*/
		
		
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
	current.release();

	return 0;
}

int compute_populationMap(VideoCapture video) {

	// @ params
	Pixel2 rectStart = Pixel2(250,150);
	Pixel2 rectEnd = Pixel2(300,200);
	int numberOfVertices = 50;

	// set up output videos
	String video_name = "populationMap";
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

	// initialize PopulationMap
	PopulationMap population = PopulationMap(rectStart, rectEnd, numberOfVertices);

	//Preload the first frame as previous frame
	video.read(frame);
	if(frame.empty()) exit(1);
	resize(frame,resized_frame,Size(XDIM,YDIM),0,0,INTER_LINEAR);
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

		Mat outImg;	// output image
		resized_frame.copyTo(outImg);

		population.runLK(u_prev, u_current, outImg);

		imshow("populationMap", outImg);
		video_output.write(outImg);
		
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

int compute_timelinesFarne(VideoCapture video) {
	

	// @ params
	Pixel2 lineStart = Pixel2(100,180);
	Pixel2 lineEnd = Pixel2(500,250);
	int numberOfVertices = 20;

	// discrete streamline seed points
	Pixel2 streampt[numberOfVertices];

	// define the distance between each vertices
	float diffX = (lineEnd.x - lineStart.x) / numberOfVertices;
	float diffY = (lineEnd.y - lineStart.y) / numberOfVertices;

	// create and push Pixel2 points
	for (int i = 0; i <= numberOfVertices; i++)
	{
		streampt[i] = Pixel2(lineStart.x + diffX * i, lineStart.y + diffY * i);
	}


	// set up output videos
	String video_name = "timelinesFarneSubtractAve";
	VideoWriter video_output( outputFileName  + ".mp4",CV_FOURCC('X','2','6','4'), 30, cv::Size(XDIM,YDIM),true);

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
		calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 20, 3, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		current = u_flow.getMat(ACCESS_READ);


		// subtructAverage(current);

		


		// draw streamlines
		Mat streamout;
		resized_frame.copyTo(streamout);

		


		get_streamlines(streamout, streamoverlay_color, streamoverlay, numberOfVertices, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		

		Mat outImg;	// output image
		resized_frame.copyTo(outImg);

		// draw edges
		circle(outImg,cvPoint(streampt[0].x,streampt[0].y),4,CV_RGB(0,0,100),-1,8,0);
		for ( int i = 0; i < (int)numberOfVertices - 1; i++ ) {
			line(outImg,cvPoint(streampt[i].x,streampt[i].y),cvPoint(streampt[i+1].x,streampt[i+1].y),CV_RGB(100,0,0),2,8,0);
			circle(outImg,cvPoint(streampt[i+1].x,streampt[i+1].y),4,CV_RGB(0,0,100),-1,8,0);
		}
		
		drawFrameCount(outImg, framecount);
		
		imshow("streamlines",outImg);
		video_output.write(outImg);
		
		
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


int compute_subtructAverageVectorWithWindow(VideoCapture video) {

	// set up output videos
	String video_name = "subtructAverageVector";
	VideoWriter video_output( outputFileName + ".mp4",CV_FOURCC('X','2','6','4'), 30, cv::Size(XDIM,YDIM),true);

	if (!video_output.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		return -1;
	}

	// OpenCV matrices to load images
	Mat frame;			// raw current frame image
	Mat resized_frame;	// resized current frame image
	Mat grayscaled_frame;			// gray scaled current frame image
	Mat current;		// Mat output velocity field of OpticalFlow
	
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
	# define MAX_STREAMLINES 40
	Pixel2 streampt[MAX_STREAMLINES];
	int streamlines = MAX_STREAMLINES/2;

	sranddev();
	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}
	// original seed point
	// streamlines = 1;
	// streampt[0] = Pixel2(300,300);

	int windowSize = 50;
	int currentBuffer = 0;
	vector<Mat> buffer;
	Mat averageCurrent = Mat::zeros(YDIM,XDIM,CV_32FC2);

	for ( int i = 0; i < windowSize; i++ )
	{
		buffer.push_back(Mat::zeros(YDIM,XDIM,CV_32FC2));
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
		// 0.5, 2, 3, 2, 15, 1.2
		calcOpticalFlowFarneback(u_prev, u_current, u_flow, 0.5, 2, 20, 3, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
		current = u_flow.getMat(ACCESS_READ);


		// draw streamlines
		Mat outImg;
		resized_frame.copyTo(outImg);

		subtructAverage(current);
		

		// draw streamlines
		Mat streamout;
		resized_frame.copyTo(streamout);
		// get_streamlines(streamout, streamoverlay_color, streamoverlay, streamlines, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		// imshow("streamlines",streamout);
		// video_output.write(streamout);


		// delete the oldest data from average
		averageCurrent -= buffer[currentBuffer] / (float)(windowSize);

		// insert new data
		buffer[currentBuffer] = current.clone();

		// add the new data to average
		averageCurrent += buffer[currentBuffer] / (float)(windowSize);

		// increment current buffer number
		currentBuffer++;
		if ( currentBuffer >= windowSize ) currentBuffer = 0;


		
		vectorToColor(averageCurrent, outImg);

		drawFrameCount(outImg, framecount);
		
		imshow("subtruct average vector",outImg);
		video_output.write(outImg);
		
		
		
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
	current.release();

	return 0;
}