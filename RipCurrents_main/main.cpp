#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name
#include <opencv2/optflow/motempl.hpp>

#include "ripcurrents.hpp"

String type2str(int type) {
  String r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


int main(int argc, char** argv )
{
	
	if(argc <2){printf("No video specified\n"); exit(0); }
	// Turn on OpenCL
	ocl::setUseOpenCL(true);

	// Set output video name
	String video_name;
	if( argc >= 2 ) video_name = argv[2];
	else video_name = "output";
	
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
	
	VideoWriter video_output( video_name + ".mov",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);
	
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
	Mat current_prev;

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

	/*for(int s = 0; s < streamlines; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}*/
	for(int x = 0; x < 10; x++){
		for(int y = 0; y < 10; y++){
			streampt[x * 10 + y] = Pixel2(XDIM * x / 10, YDIM * y / 10);
		}
	}
	namedWindow("streamlines", WINDOW_AUTOSIZE );
	
	
	int framecount; //Generic iterator for main loop.
	
	
	//Preload a frame
	video.read(frame);
	if(frame.empty()){exit(1);}
	resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_AREA);
	cvtColor(subframe,f1,COLOR_BGR2GRAY);
	f1.copyTo(u_f2);

	// stop loop with key
	int k = -1;


	std::vector<Mat>buffer;
	for ( int i = 0; i < BUFFER_FRAME; i++ ) {
		buffer.push_back(Mat::zeros(YDIM,XDIM,CV_32FC2));
	}
	int update_ith_buffer = 0;
	Mat average = Mat::zeros(YDIM,XDIM,CV_32FC2);
	Mat average_color = Mat::zeros(Size(XDIM, YDIM), CV_8UC3);

	float max_displacement = 0.000001;
	

	// create 2d array for grid
	double ** grid = new double*[GRID_COUNT];
	for ( int i = 0; i < GRID_COUNT; i++ )
		grid[i] = new double[GRID_COUNT];
	
	// number of rows and cols in each grid
	int grid_col_num = (int)(XDIM/GRID_COUNT);
	int grid_row_num = (int)(YDIM/GRID_COUNT);

	timediff();
	for( framecount = 1; true; framecount++){

		

		video.read(frame);
	
		printf("Frames read: %d\n",framecount);


		
		if(frame.empty()){break;}


		//Resize
		resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(subframe,f1,COLOR_BGR2GRAY);
		
		//Move to GPU (if possible), compute flow, move back
		f1.copyTo(u_f1);
		//Parameters are tweakable
		calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN); //Give to GPU, possibly
		flow_raw = u_flow.getMat(ACCESS_READ); //Tell GPU to give it back
		Mat current = flow_raw;

		/*
		// stabilize with corner tracking
		if(framecount > 2) stabilizer(current, current_prev);
		current.copyTo(current_prev);
		*/
		

		
		
		// global orientation of entire image
		/*Mat hist_gray;
		globalOrientation(u_f1, u_f2, hist_gray);
		imshow("angle", hist_gray);
		video_output.write(hist_gray);
		*/
		

		u_f1.copyTo(u_f2);

		

		//Simulate the movement of particles in the flow field.
		streamlines_mat.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {
			streamline_field(&pixel, streamlines_distance.ptr<float>(position[0],position[1]), position[1],position[0], current, 2, 1,UPPER,prop_above_upper);
		});


		//average_vector();
		// uppdate buffer range 0 <= x < BUFFER_FRAME
		if ( update_ith_buffer >= BUFFER_FRAME ) update_ith_buffer -= BUFFER_FRAME;

		// subtract old buffer data from average
		average -= buffer[update_ith_buffer] / BUFFER_FRAME;
		buffer[update_ith_buffer] = Mat::zeros(YDIM,XDIM,CV_32FC2);
		// get new buffer
		buffer[update_ith_buffer].forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void{
			get_delta(&pixel, streamlines_distance.ptr<float>(position[0],position[1]), position[1],position[0], current, 2, 1,UPPER,prop_above_upper);
		});

		// add new buffer to average
		average += buffer[update_ith_buffer] / BUFFER_FRAME;

		float global_theta = 0;
		float global_magnitude = 0;

		int grid_row = 1, grid_col = 1;

		for ( int i = 0; i < GRID_COUNT; i++ ) {
			for ( int j = 0; j < GRID_COUNT; j++ ) {
				grid[i][j] = 0;
			}
		}

		// store vector data of average
		int co = 0;
		for ( int row = 0; row < average.rows; row++ ) {
			Pixel2* ptr = average.ptr<Pixel2>(row, 0);
			Pixelc* ptr2 = average_color.ptr<Pixelc>(row, 0);

			if ( row >= grid_row_num * grid_row)
				grid_row++;
			
			grid_col = 1;

			for ( int col = 0; col < average.cols; col++ ) {
				float theta = atan2(ptr->y, ptr->x)*180/M_PI;	// find angle
				theta += theta < 0 ? 360 : 0;	// enforce strict positive angle
				
				// store vector data
				ptr2->x = theta / 2;
				ptr2->y = 255;
				ptr2->z = sqrt(ptr->x * ptr->x + ptr->y * ptr->y)*255/max_displacement;
				if ( ptr2->z < 20 ) ptr2->z = 0;

				// store the previous max to maxmin next frame
				if ( sqrt(ptr->x * ptr->x + ptr->y * ptr->y) > max_displacement ) max_displacement = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

				global_theta += ptr2->x * ptr2->z;
				global_magnitude += ptr2->z;

				if ( col >= grid_col_num * grid_col)
					grid_col++;

				// add the vector to the corresponding grid
				grid[grid_row-1][grid_col-1] += theta;
				if (grid_col == 5 && grid_row == 5){
					co++;
				}

				ptr++;
				ptr2++;
			}
		}

		// draw global orientation arrow
		circle(average_color, Point((int)(XDIM/2), (int)(YDIM/2)), 3, Scalar(0, 215, 255), CV_FILLED, 16, 0);
		double global_angle_rad = global_theta * 2 / global_magnitude * M_PI / 180;
		arrowedLine(average_color, Point((int)(XDIM / 2), (int)(YDIM / 2)), 
			Point((int)(XDIM / 2 + cos(global_angle_rad) * 10), (int)(YDIM / 2 + sin(global_angle_rad) * 50)),
			Scalar(0, 215, 255), 2, 16, 0, 0.2);

		// show as hsv format
		cvtColor(average_color, average_color, CV_HSV2BGR);

		// draw arrows for each grid
		for ( int row = 1; row < GRID_COUNT; row++ ){
			for ( int col = 1; col < GRID_COUNT; col++ ){
				double angle_deg = grid[row][col] / co;
				double angle_rad = angle_deg  * M_PI / 180;
				// find in-between angle
				double angle_between = min(abs(global_angle_rad - angle_rad), 2*M_PI-abs(global_angle_rad - angle_rad));
				if ( angle_between > M_PI * 0.6 ) {
					circle(average_color, Point(col * grid_col_num, row * grid_row_num), 1, Scalar(0, 215, 0), CV_FILLED, 16, 0);
					arrowedLine(average_color, Point(col * grid_col_num, row * grid_row_num), 
						Point((int)(col * grid_col_num + cos(angle_rad) * 10), (int)(row * grid_row_num + sin(angle_rad) * 10)),
						Scalar(0, 215, 0), 1, 16, 0, 0.4);
				}
			}
		}


		update_ith_buffer++;

		imshow("average vector", average_color);
		video_output.write(average_color);

		
		Mat streamfield;
		split(streamlines_mat,splitarr);
		magnitude(splitarr[0],splitarr[1],streamfield);
		
		// How far it moved
		streamline_displacement(streamfield, streamoverlay_color);
		//imshow("streamline displacement",streamoverlay_color);
		//video_output.write(streamoverlay_color);

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
		imshow("streamlines",streamout);
		//video_output.write(streamout);

		
		
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

		//create_flow(current, waterclass, accumulator2, UPPER, MID, LOWER, UPPER2d);
		//cvtColor(current,current,CV_HSV2BGR);
		//imshow("flow",current);

		
		//Mat out = Mat::zeros(Size(XDIM, YDIM), CV_32FC3);
		//Mat outmask = Mat::zeros(Size(XDIM, YDIM), CV_8UC1);

		//create_accumulationbuffer(accumulator, accumulator2, out, outmask, framecount);
		//imshow("accumulationbuffer",out);

		
		//create_edges(outmask);
		//imshow("edges",outmask);

		
		//create_output(subframe, outmask);
		//imshow("output",subframe);

		// end with Esc key on any window
		int c = waitKey(1);
		if ( c == 27) break;

		// stop and restart with any key
		if ( c != -1 && c != 27 ) {
			waitKey(0);
		}
	}

	//Clean up
	flow_raw.release();
	
	//waitKey(0);
	video.release();
	video_output.release();

	// closed all windows
	destroyAllWindows();
	
	return 0;
}
