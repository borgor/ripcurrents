#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name

#include "ripcurrents.hpp"


int main(int argc, char** argv )
{
	
	if(argc <2){printf("No video specified\n"); exit(0); }
	// Turn on OpenCL
	ocl::setUseOpenCL(true);
	
	//Video I/O
	VideoCapture video;
	if(*argv[1] == (char)'-'){
		video = VideoCapture(0);
		if (!video.isOpened())
		{
			std::cout << "!!! Input video could not be opened" << std::endl;
			exit(-1);
		}
	} else {
		video = VideoCapture(argv[1]);
		if (!video.isOpened())
		{
			std::cout << "!!! Input video could not be opened" << std::endl;
			exit(-1);
		}
	}
	
	// Set up for output videos
	
	VideoWriter video_output("video_output.mp4",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);
	
	if (!video_output.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		exit(-1);
	}
	
	

	double time_farneback = 0;
	double time_polar = 0;
	double time_threshold = 0;
	double time_overlay = 0;
	double time_erosion = 0;
	double time_codec = 0;
	double time_stream = 0;
	int frames_read = 0;
	
	

//	float scalex = XDIM/video.get(CAP_PROP_FRAME_WIDTH);
//	float scaley = YDIM/video.get(CAP_PROP_FRAME_HEIGHT);
	int totalframes = (int) video.get(CAP_PROP_FRAME_COUNT);
	
	//A lot of matrices/frames
	Mat save;
	Mat frame,f1;
	Mat subframe;
	Mat resized;
	Mat flow_raw;
	Mat flow;

	//OpenCL/GPU matrices
	UMat u_flow;
	UMat u_f1,u_f2;
	

	
	//Zero out accumulators
	Mat accumulator = Mat::zeros(Size(XDIM,YDIM), CV_32FC3);
	
	Mat out = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);
	
	Mat splitarr[2];
	
	

	//Some thresholds to mask out any remaining jitter, and strong waves. Don't know how to calculate them at runtime, so they're arbitrary.
	float LOWER =  0.2;
	float MID  = .5;


	
	int hist[HIST_BINS] = {0}; //histogram
	int histsum = 0;
	float UPPER = 100.0; //UPPER can be determined programmatically
	
	int hist2d[HIST_DIRECTIONS][HIST_BINS] = {{0}};
	int histsum2d[HIST_DIRECTIONS] = {0};
	float UPPER2d[HIST_DIRECTIONS] = {0};
	float prop_above_upper[HIST_DIRECTIONS] = {0};
	
	
	
	
	sranddev();
	Mat streamoverlay = Mat::zeros(Size(XDIM, YDIM), CV_8UC1);
	Mat streamoverlay_color = Mat::zeros(Size(XDIM, YDIM), CV_8UC3);

	//initialize streamline scalar field
	Mat streamlines_mat = Mat::zeros(YDIM,XDIM,CV_32FC2); //Track displacement from initial point
	Mat streamlines_distance = Mat::zeros(YDIM,XDIM,CV_32FC1); //Track total distance traveled
	
	 //Code for discrete streamline initialization
	# define MAX_STREAMLINES 500
	Pixel2 streampt[MAX_STREAMLINES];
	int streamlines = MAX_STREAMLINES/2;

	for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}
	namedWindow("streamlines", WINDOW_AUTOSIZE );
	
	
	int framecount; //Generic iterator for main loop.
	
	
	//Preload a frame
	video.read(frame);
	if(frame.empty()){exit(1);}
	resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_AREA);
	cvtColor(subframe,f1,COLOR_BGR2GRAY);
	f1.copyTo(u_f2);




	timediff();
	for( framecount = 1; true; framecount++){

		

		video.read(frame);
	
		printf("Frames read: %d\n",framecount);


		
		if(frame.empty()){break;}
		time_codec += timediff();


		//Resize
		resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(subframe,f1,COLOR_BGR2GRAY);
		
		//Move to GPU (if possible), compute flow, move back
		f1.copyTo(u_f1);
		//Parameters are tweakable
		calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 2, 3, 2, 15, 1.2, 0); //Give to GPU, possibly
		u_f1.copyTo(u_f2);
		flow_raw = u_flow.getMat(ACCESS_READ); //Tell GPU to give it back
		
		
		
		Mat current = flow_raw;
		
		time_farneback += timediff();
		
		

		//Simulate the movement of particles in the flow field.
		
		streamlines_mat.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {
			streamline_field(&pixel, streamlines_distance.ptr<float>(position[0],position[1]), position[1],position[0], current, 2, 1,UPPER,prop_above_upper);
		});
		
		Mat streamfield;
		split(streamlines_mat,splitarr);
		magnitude(splitarr[0],splitarr[1],streamfield);
		
		// How far it moved
		streamline_displacement(streamfield, streamoverlay_color);
		imshow("streamline displacement",streamoverlay_color);

		// How far it has moved
		streamline_total_motion(streamlines_distance, streamoverlay_color);
		//imshow("streamline total motion",streamoverlay_color);

		// Ratio of displacement / motion
		streamline_ratio(streamfield, streamlines_distance, streamoverlay_color);
		//imshow("streamline displacement/motion ratio",streamoverlay_color);
		
		
		
		Mat streamline_density = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);
		streamline_positions(streamlines_mat, streamline_density);
		//imshow("streamline positions",streamline_density);

		
		
		//Discrete,drawable streamlines handled here
		// creates a copy of current frame
		Mat streamout;
		subframe.copyTo(streamout);
		get_streamlines(streamout, streamoverlay_color, streamoverlay, streamlines, streampt, framecount, totalframes, current, UPPER, prop_above_upper);
		//imshow("streamlines",streamout);

		
		
		//convert the x,y current flow field into angle,magnitude form.
		//Specifically, angle,magnitude,magnitude, as it is later displayed with HSV
		//This is more interesting to analyze
		split(current,splitarr);
		Mat combine[3];
		cartToPolar(splitarr[0], splitarr[1], combine[2], combine[0],true);
		combine[1] = combine[2];
		merge(combine,3,current);

		//Construct histograms to get thresholds
		//Figure out what "slow" or "fast" is
		create_histogram(current,  hist, histsum, hist2d, histsum2d, UPPER, UPPER2d, prop_above_upper);
		//display_histogram(hist2d,histsum2d,UPPER2d, UPPER,prop_above_upper);
		
		
		
		Mat accumulator2 = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);
		Mat waterclass = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);

		create_flow(current, waterclass, accumulator2, UPPER, MID, LOWER, UPPER2d);
		cvtColor(current,current,CV_HSV2BGR);
		//imshow("flow",current);

		
		Mat out = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);
		Mat outmask = Mat::zeros(Size(XDIM, YDIM), CV_8UC1);

		create_accumulationbuffer(accumulator, accumulator2, out, outmask, framecount);
		//imshow("accumulationbuffer",out);

		
		create_edges(outmask);
		//imshow("edges",outmask);

		
		create_output(subframe, outmask);
		imshow("output",subframe);

		waitKey(1);
	}
	//Clean up
	
	flow_raw.release();
	
	//waitKey(0);
	video.release();
	video_output.release();
	
	waitKey(0);
	
	return 0;
}
