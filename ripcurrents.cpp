#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name


#include "pathlines.h"


#define STABILIZE 5	//Size of buffer for stabilizing video


#define XDIM 640   //Dimensions to resize to
#define YDIM 480




using namespace cv;

typedef cv::Point3_<uchar> Pixelc;
typedef cv::Point_<float> Pixel2;
typedef cv::Point3_<float> Pixel3;

void wheel(); //color wheel function
int rip_main(cv::VideoCapture video, cv::VideoWriter video_out);


double timediff(){
	static struct timeval oldtime;
	struct timeval newtime;
	gettimeofday(&newtime,NULL);
	double diff = (newtime.tv_sec - oldtime.tv_sec) + (newtime.tv_usec - oldtime.tv_usec)/1000000.0;
	gettimeofday(&oldtime,NULL);
	return diff;
}



int main(int argc, char** argv )
{
	
	if(argc <2){printf("No video specified\n");wheel(); exit(0); }
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
	
	
	VideoWriter video_out("video_out.avi",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);
	
	if (!video_out.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		exit(-1);
	}
	
	rip_main(video, video_out);
	
}

int rip_main(cv::VideoCapture video, cv::VideoWriter video_out){
	double time_farneback = 0;
	double time_polar = 0;
	double time_threshold = 0;
	double time_overlay = 0;
	double time_erosion = 0;
	double time_codec = 0;
	double time_stream = 0;
	int frames_read = 0;
	
	

	float scalex = XDIM/video.get(CAP_PROP_FRAME_WIDTH);
	float scaley = YDIM/video.get(CAP_PROP_FRAME_HEIGHT);
	int totalframes = (int) video.get(CAP_PROP_FRAME_COUNT);
	
	//A lot of matrices/frames
	Mat save;
	Mat frame,f2;
	Mat subframe;
	Mat resized;
	Mat flow_raw;
	Mat flow;
	Mat stable[STABILIZE];

	//OpenCL/GPU matrices
	UMat u_flow;
	UMat u_f1,u_f2;
	
	/*
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2();
	Mat fgMaskMOG2;
	Mat MOG_accumulator = Mat::zeros(YDIM,XDIM,CV_32FC1);
	*/
	
	//Zero out accumulators
	Mat accumulator = Mat::zeros(YDIM, XDIM, CV_32FC3);
	for(int j = 0 ; j< STABILIZE; j++){stable[j] = Mat::zeros(YDIM, XDIM, CV_32FC2);}
	
	Mat out = Mat::zeros(YDIM, XDIM, CV_32FC3);
	
	Mat splitarr[2];
	

	//Output windows
	//namedWindow("Rip Current Detector", WINDOW_AUTOSIZE );
	//namedWindow("Flow", WINDOW_AUTOSIZE );
	//namedWindow("Classifier", WINDOW_AUTOSIZE );
	//namedWindow("Accumulator", WINDOW_AUTOSIZE );
	
	
	#define HIST_BINS 100 //Number of bins for finding thresholds
	#define HIST_DIRECTIONS 36 //Number of 2d histogram directions
	//Some thresholds to mask out any remaining jitter, and strong waves. Don't know how to calculate them.
	float LOWER =  0.2;
	float MID  = .5;


	
	int hist[HIST_BINS] = {0}; //histogram
	int histsum = 0;
	float UPPER = 100.0; //UPPER can be determined programmatically
	
	int hist2d[HIST_DIRECTIONS][HIST_BINS] = {0};
	int histsum2d[HIST_DIRECTIONS] = {0};
	float UPPER2d[HIST_DIRECTIONS] = {0};
	
	
	
	
	int framecount; //Generic iterator for main loop.
	
	
	int turn = 0;  //Alternator
	
	//Preload a frame
	video.read(frame);
	if(frame.empty()){exit(1);}
	resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_AREA);
	cvtColor(subframe,f2,COLOR_BGR2GRAY);
	f2.copyTo(u_f1);


# define STREAMLINES 250
	
	sranddev();
	Mat streamoverlay = Mat::zeros(YDIM, XDIM, CV_8UC1);
	Mat streamoverlay_color = Mat::zeros(YDIM, XDIM, CV_8UC3);
	Pixel2 streampt[STREAMLINES];
	for(int s = 0; s < STREAMLINES; s++){
		streampt[s] = Pixel2(rand()%XDIM,rand()%YDIM);
	}


	timediff();
	for( framecount = 1; true; framecount++){

		
		
		//video.read(frame);
		//video.read(frame); //skip frames for speed
		video.read(frame);
		frames_read+=1;
		printf("Frames read: %d\n",frames_read);
		
		if(frame.empty()){break;}
		time_codec += timediff();


		//Resize, turn to gray.
		resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(subframe,f2,COLOR_BGR2GRAY);
		if(turn){
			f2.copyTo(u_f1);
			calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 2, 3, 2, 15, 1.2, 0); //Give to GPU
			//printf("tick\n");
		}else{
			f2.copyTo(u_f2);
			calcOpticalFlowFarneback(u_f1,u_f2, u_flow, 0.5, 2, 3, 2, 15, 1.2, 0);
			//printf("tock\n");
		}
		turn = !turn;
		
		
		
		flow_raw = u_flow.getMat(ACCESS_READ); //Tell GPU to give it back
		
		for(int j = 0; j<STABILIZE; j++){
			add(flow_raw,stable[j],stable[j]);
		}
		
		
		
		Mat current = stable[framecount%STABILIZE]*(1.0/STABILIZE);
		
		time_farneback += timediff();
		

		for(int s = 0; s < STREAMLINES; s++){
			streamline(streampt+s, Scalar(framecount*(255.0/totalframes)), current, streamoverlay, 2, 1);
		}
		
		applyColorMap(streamoverlay, streamoverlay_color, COLORMAP_RAINBOW);
		Mat streamout;
		subframe.copyTo(streamout);
		add(streamoverlay_color, streamout, streamout, streamoverlay, -1);
		imshow("stream",streamout);
		



		time_stream+= timediff();

		/*
		pMOG2->apply(subframe, fgMaskMOG2);
		imshow("fg",fgMaskMOG2);
		Mat foo;
		pMOG2->getBackgroundImage(foo);
		imshow("background",foo);
		//add(fgMaskMOG2,MOG_accumulator,MOG_accumulator,noArray(),MOG_accumulator.depth());
		//time_mog += timediff();
		*/
		
		//convert the x,y current flow field into angle,magnitude form.
		//Specifically, angle,magnitude,magnitude, as it is later displayed with HSV
		split(current,splitarr);
		cartToPolar(splitarr[0], splitarr[1], splitarr[1], splitarr[0],true);
		Mat combine[3] = {splitarr[0],splitarr[1],splitarr[1]};
		merge(combine,3,current);
		
	
		//imshow("pathline",visibleflow+streamoverlay);
		
		time_polar += timediff();
		
		
		int resolution = 10;
		//Fill histogram
		
		//update to 2D histogram
		
		for (int y = 0; y < YDIM; y++) {
			Pixel3* ptr = current.ptr<Pixel3>(y, 0);
			const Pixel3* ptr_end = ptr + (int)XDIM;
			for (int x = 0 ; ptr != ptr_end; ++ptr, x++) {
				int bin = (ptr->y) * resolution;
				int angle = (ptr->x * HIST_DIRECTIONS)/ 360; //order matters, truncation
				if(bin < HIST_BINS &&  bin >= 0){
					hist[bin]++; histsum++;
					hist2d[angle][bin]++; histsum2d[angle]++;
				}
				
			}
		}
		
		//Use histogram to create overall threshold
		int threshsum = 0;
		int bin = HIST_BINS-1;
		while(threshsum < (histsum*.03)){
			threshsum += hist[bin];
			bin--;
		}
		UPPER = bin/float(resolution);
		
		
		//As above, but per-direction
		for(int angle = 0; angle < HIST_DIRECTIONS; angle++){
			int threshsum = 0;
			int bin = HIST_BINS-1;
			while(threshsum < (histsum2d[angle]*.03)){
				threshsum += hist2d[angle][bin];
				bin--;
			}
			UPPER2d[angle] = bin/float(resolution);
			if(UPPER2d[angle] < 0.01) {UPPER2d[angle] = 0.01;}//avoid division by 0
			printf("Angle %d; frequency: %d; upper: %f\n",angle,histsum2d[angle], UPPER2d[angle]);

		}
		

		

		Mat accumulator2 = Mat::zeros(YDIM, XDIM, CV_32FC3);
		Mat waterclass = Mat::zeros(YDIM, XDIM, CV_32FC3);
		

		//Classify
		current.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
			
			Pixel3* classptr = waterclass.ptr<Pixel3>(position[0],position[1]);
			Pixel3 * pt = accumulator2.ptr<Pixel3>(position[0],position[1]);
			int angle = (pixel.x * HIST_DIRECTIONS)/ 360; //order matters, truncation
			float val = pixel.z ;
			
			//Load classifier and overlay
			if(val > UPPER){classptr->x = .5; pt->x++;}else{
				if(val > MID){classptr->z = 1;}else{
					if(val > LOWER){classptr->z = .5;}else{
						 {classptr->y = .5;}
					}
				}
			}
			
			//Rescale for display
			//pixel.z = 1;
			pixel.z = val/UPPER2d[angle];
			if(pixel.z > 1){
				pixel.y = 1;
			} else {
				pixel.y = .7;
			}
			
			
		});

		
		cvtColor(current,current,CV_HSV2BGR);
		imshow("flow",current);
		

		
		
		time_threshold += timediff();
		
		//accumulator accumulates waves
		if(framecount > 30){
			add(accumulator2,accumulator,accumulator);
		}
		

		Mat out = Mat::zeros(YDIM, XDIM, CV_32FC3);
		Mat outmask = Mat::zeros(YDIM, XDIM, CV_8UC1);
		
		
		//Visualize accumulation buffer. Thresholds need tweaking
		out.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
			Pixel3* accptr = accumulator.ptr<Pixel3>(position[0],position[1]);
			uchar* maskptr = outmask.ptr<uchar>(position[0],position[1]);
			
			int val = accptr->x;
			if(val > .1 * framecount){
				if(val < .2 * framecount){
					pixel.z = 1;
				}else{
					pixel.x = 1;
				}
			}else{
					pixel.y = .5;
					*maskptr = 255;
			}
		});
		
		//Use morphological operations to reduce noise, find edges
		
		
		

		//imshow("accumulationbuffer",out);
		
		Mat morph_window;
		
		//imshow("original",outmask);
		
		
		/*  //Option 1: mess with fill
		Mat erode1, erode2;
		
		morph_window = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
		dilate(outmask, erode2, morph_window);
		
		morph_window = getStructuringElement(MORPH_ELLIPSE, Size(30,30));
		morphologyEx( outmask, erode1, 2, morph_window ); //Morphology operation #2: open: erode, dilate
		
		imshow("eroded",erode1);
		
		morph_window = getStructuringElement(MORPH_ELLIPSE, Size(2,2));
		morphologyEx( erode2, outmask, 4, morph_window ); //Morphology operation #4: gradient (dilation - erosion)
		
		
		
		imshow("edges",outmask);
		
		floodFill(outmask, Point(0,0), 255);
		imshow("fill",outmask);
		
		outmask -= erode1;
		
		imshow("cleaned",outmask);
		*/
		
		//Option 2: just edges
		
		morph_window = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
		dilate(outmask, outmask, morph_window);
		morphologyEx( outmask, outmask, 4, morph_window ); //Morphology operation #4: gradient (dilation - erosion)
		imshow("edges",outmask);
		
		
		time_erosion += timediff();
		
		
	
		/*
		Mat edges, edges2;
		out.convertTo(edges,CV_8UC1,255);
		cvtColor(edges,edges,CV_BGR2GRAY);
		Canny(edges,edges2,10,50,5,false);
		imshow("edges",edges2);
		
		time_edges += timediff();
		*/
		
		
		Mat overlay = Mat::zeros(YDIM, XDIM, CV_8UC3);
		
		//Find green surrounded by red in accumulator image, create overlay
#define localwin 20
		/*
		for (int y = 0; y < YDIM- localwin*2; y+=localwin) {
			for (int x = 0 ; x < XDIM - localwin*2; x+=localwin) {
				int hisum = 0; int losum = 0;
				for(int k = 0; k < localwin*2; k++){
					for(int j = 0; j<localwin*2; j++){
						if(out.ptr<Pixel3>(y+j, x+k)->z){hisum++;}
						if(out.ptr<Pixel3>(y+j, x+k)->y){losum++;}
					}
				}
				if(hisum > localwin*localwin/1.5 && losum > localwin*localwin/1.5){
					//printf("%d %d\n",hisum,losum);
					for(int k = 0; k < localwin*2; k++){
						for(int j = 0; j<localwin*2; j++){
							if(out.ptr<Pixel3>(y+j, x+k)->y){overlay.ptr<Pixelc>(y+j, x+k)->z ++;}
						}
					}
				}
			}
		}
		 */

		
		//Combine overlay and original
		if(true/*framecount>90*/){
			subframe.forEach<Pixelc>([&](Pixelc& pixel, const int position[]) -> void {
				uchar over =  *outmask.ptr<uchar>(position[0],position[1]);
				uchar stream =  *streamoverlay.ptr<uchar>(position[0],position[1]);
				if(over){
					pixel.z = 255;
				}
				//if(stream){
				//	pixel.x = 255;
				//	pixel.y = 0;
				//	pixel.z = 255;
				//}
				
			});
		}
	
		time_overlay += timediff();
		imshow("output",subframe);
		
		video_out.write(streamout);

		waitKey(1);
		stable[framecount%STABILIZE] = Mat::zeros(YDIM, XDIM, CV_32FC2);
		
		timediff();
		
	}
	printf("Frames read: %d\n",frames_read);
	printf("Time spent on farneback: %f\n",time_farneback);
	printf("Time spent on polar coordinates: %f\n",time_polar);
	printf("Time spent on thresholds: %f\n",time_threshold);
	printf("Time spent on overlay: %f\n",time_overlay);
	printf("Time spent on erosion: %f\n",time_erosion);
	printf("Time spent on codec: %f\n",time_codec);
	printf("Time spent on pathlines: %f\n",time_stream);
	
	//Clean up
	
	flow_raw.release();
	
	//waitKey(0);
	video.release();
	video_out.release();
	
	return 0;
}



void wheel(){ //Display the color wheel
	
	namedWindow("Color Wheel", WINDOW_AUTOSIZE );
	
	Mat foo = Mat::ones(YDIM, YDIM, CV_32FC3);
	
	foo.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
		
		float tx = (position[1]-240)/240.0;
		float ty = (position[0]-240)/240.0;
		
		
		int bin = (int) floor(atan(ty/tx)/M_PI  * 18  );//Begins to calculate the angle as an integer between 1 and 35.
		
		
		
		if(ty>0){
			if(tx>0){
				bin = bin;
			}else{
				bin = 18 + bin;
			}
		}else{
			
			if(tx<0){
				bin = bin+18;
			}else{
				bin = 36 + bin;
			}
		}
		pixel.x = bin * 10;
		
		
		
		float d = sqrt(tx*tx + ty *ty);
		
		
		pixel.y = d > 1 ? 0 : d;
		pixel.z = d > 1 ? 0 : 1;
		//pixel.z = pixel.x;
		
	});
	
	
	cvtColor(foo,foo,CV_HSV2BGR);
	imshow("Color Wheel",foo);
	
	waitKey(0);
	
	exit(0);

	
}

