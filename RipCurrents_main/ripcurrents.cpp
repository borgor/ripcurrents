#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name







#define XDIM 640   //Dimensions to resize to
#define YDIM 480

#define HIST_BINS 50 //Number of bins for finding thresholds
#define HIST_DIRECTIONS 36 //Number of 2d histogram directions
#define HIST_RESOLUTION 20

using namespace cv;

typedef cv::Point3_<uchar> Pixelc;
typedef cv::Point_<float> Pixel2;
typedef cv::Point3_<float> Pixel3;

void wheel(); //color wheel function

void display_histogram(int hist2d[HIST_DIRECTIONS][HIST_BINS], int histsum2d[HIST_DIRECTIONS],
					float UPPER2d[HIST_DIRECTIONS], float UPPER, float prop_above_upper[HIST_DIRECTIONS]); //display the 2d histogram, using code from the wheel
//give it a pointer to the array, make sure indexing works

void streamline(Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]);

void streamline_field(Pixel2 * pt, float* distancetraveled, int xoffset, int yoffset, cv::Mat flow, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]);

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
	
	
	VideoWriter video_streamlines("video_streamlines.avi",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);
	
	if (!video_streamlines.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		exit(-1);
	}
	
	VideoWriter video_borders("video_borders.avi",CV_FOURCC('M','J','P','G'), 10, cv::Size(XDIM,YDIM),true);
	
	if (!video_borders.isOpened())
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
	
	/*
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2();
	Mat fgMaskMOG2;
	Mat MOG_accumulator = Mat::zeros(YDIM,XDIM,CV_32FC1);
	*/
	
	//Zero out accumulators
	Mat accumulator = Mat::zeros(YDIM, XDIM, CV_32FC3);
	
	Mat out = Mat::zeros(YDIM, XDIM, CV_32FC3);
	
	Mat splitarr[2];
	

	//Output windows
	//namedWindow("Rip Current Detector", WINDOW_AUTOSIZE );
	//namedWindow("Flow", WINDOW_AUTOSIZE );
	//namedWindow("Classifier", WINDOW_AUTOSIZE );
	//namedWindow("Accumulator", WINDOW_AUTOSIZE );
	
	

	//Some thresholds to mask out any remaining jitter, and strong waves. Don't know how to calculate them.
	float LOWER =  0.2;
	float MID  = .5;


	
	int hist[HIST_BINS] = {0}; //histogram
	int histsum = 0;
	float UPPER = 100.0; //UPPER can be determined programmatically
	
	int hist2d[HIST_DIRECTIONS][HIST_BINS] = {0};
	int histsum2d[HIST_DIRECTIONS] = {0};
	float UPPER2d[HIST_DIRECTIONS] = {0};
	float prop_above_upper[HIST_DIRECTIONS] = {0};
	
	
	
	
	sranddev();
	Mat streamoverlay = Mat::zeros(YDIM, XDIM, CV_8UC1);
	Mat streamoverlay_color = Mat::zeros(YDIM, XDIM, CV_8UC3);

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
	
	
	int turn = 0;  //Alternator
	
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


		//Resize, turn to gray.
		resize(frame,subframe,Size(XDIM,YDIM),0,0,INTER_LINEAR);
		cvtColor(subframe,f1,COLOR_BGR2GRAY);
		f1.copyTo(u_f1);
		calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 2, 3, 2, 15, 1.2, 0); //Give to GPU, possibly
		u_f1.copyTo(u_f2);
		
		
		
		
		flow_raw = u_flow.getMat(ACCESS_READ); //Tell GPU to give it back
		
		
		
		Mat current = flow_raw;
		
		time_farneback += timediff();
		
		
	if(framecount>100)
	{
		
		
		streamlines_mat.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {
			streamline_field(&pixel, streamlines_distance.ptr<float>(position[0],position[1]), position[1],position[0], current, 2, 1,UPPER,prop_above_upper);
		});
		Mat streamfield;
		split(streamlines_mat,splitarr);
		magnitude(splitarr[0],splitarr[1],streamfield);
		streamfield.convertTo(streamoverlay_color,CV_8UC1,500/sqrt(XDIM*YDIM));
		applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
		imshow("streamline field",streamoverlay_color);
		streamlines_distance.convertTo(streamoverlay_color,CV_8UC1,500/sqrt(XDIM*YDIM));
		applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
		imshow("streamline total motion",streamoverlay_color);
		divide(streamfield,streamlines_distance,streamoverlay_color);
		streamoverlay_color.convertTo(streamoverlay_color,CV_8UC1,125);
		applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
		imshow("streamline displacement/motion ratio",streamoverlay_color);
		
		for(int s = 0; s < streamlines; s++){
			streamline(streampt+s, Scalar(framecount*(255.0/totalframes)), current, streamoverlay, 2, 1,UPPER,prop_above_upper);
		}
	}
		

		applyColorMap(streamoverlay, streamoverlay_color, COLORMAP_RAINBOW);
		Mat streamout;
		subframe.copyTo(streamout);
		add(streamoverlay_color, streamout, streamout, streamoverlay, -1);
		imshow("streamlines",streamout);
		



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
		Mat combine[3];
		cartToPolar(splitarr[0], splitarr[1], combine[2], combine[0],true);
		combine[1] = combine[2];
		merge(combine,3,current);
		
	
		//imshow("pathline",visibleflow+streamoverlay);
		
		time_polar += timediff();
		
		
		
		//Fill histogram
		
		/*
		hist = {0};
		histsum = 0;
		hist2d = {0};
		histsum2d = {0};
		UPPER2d = {0};
		prop_above_upper = {0};
		*/
		
		
		for (int y = 0; y < YDIM; y++) {
			Pixel3* ptr = current.ptr<Pixel3>(y, 0);
			const Pixel3* ptr_end = ptr + (int)XDIM;
			for (int x = 0 ; ptr != ptr_end; ++ptr, x++) {
				int bin = (ptr->y) * HIST_RESOLUTION;
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
		while(threshsum < (histsum*.05)){
			threshsum += hist[bin];
			bin--;
		}
		UPPER = bin/float(HIST_RESOLUTION);
		int targetbin = bin;
		
		
		//As above, but per-direction
		for(int angle = 0; angle < HIST_DIRECTIONS; angle++){
			int threshsum2 = 0;
			int bin = HIST_BINS-1;
			while(threshsum2 < (histsum2d[angle]*.05)){
				threshsum2 += hist2d[angle][bin];
				bin--;
			}
			UPPER2d[angle] = bin/float(HIST_RESOLUTION);
			if(UPPER2d[angle] < 0.01) {UPPER2d[angle] = 0.01;}//avoid division by 0
			
			int threshsum3 = 0;
			bin = HIST_BINS-1;
			while(bin > targetbin){
				threshsum3 += hist2d[angle][bin];
				bin--;
			}
			prop_above_upper[angle] = ((float)threshsum3)/threshsum;
			
			
			//printf("Angle %d; prop_above_upper: %f\n",angle,prop_above_upper[angle]);
		}
		
		display_histogram(hist2d,histsum2d,UPPER2d, UPPER,prop_above_upper);
		

		Mat accumulator2 = Mat::zeros(YDIM, XDIM, CV_32FC3);
		Mat waterclass = Mat::zeros(YDIM, XDIM, CV_32FC3);

		/*
		Mat whitewater = Mat::zeros(YDIM,XDIM, CV_8UC1);
		
		subframe.forEach<Pixelc>([&](Pixelc& pixel, const int position[]) -> void {
			uchar* wtptr = whitewater.ptr<uchar>(position[0],position[1]);
			if(pixel.x > 150 && pixel.y > 150 && pixel.z > 150){ 
				*wtptr = 255;
			}
	});
		imshow("white pixels",whitewater);
		*/

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
		
		
		

		imshow("accumulationbuffer",out);
		
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
				//uchar stream =  *streamoverlay.ptr<uchar>(position[0],position[1]);
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
	
		//time_overlay += timediff();
		imshow("output",subframe);
		
		video_streamlines.write(streamout);
		video_borders.write(subframe);

		waitKey(1);
		
		//timediff();
		
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
	video_streamlines.release();
	video_borders.release();
	
	waitKey(0);
	
	return 0;
}

void display_histogram(int hist2d[HIST_DIRECTIONS][HIST_BINS],int histsum2d[HIST_DIRECTIONS]
					,float UPPER2d[HIST_DIRECTIONS], float UPPER, float prop_above_upper[HIST_DIRECTIONS]){
	namedWindow("Color Histogram", WINDOW_AUTOSIZE );
	
	Mat foo = Mat::ones(480, 480, CV_32FC3);
	
	foo.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
		
		
		float tx = (position[1]-240.0)/240.0;
		float ty = (position[0]-240.0)/240.0;
		
		float theta = atan2(ty,tx)*180/M_PI;//find angle
		theta += theta < 0 ? 360 : 0; //enforce strict positive angle
		float r = sqrt(tx*tx + ty*ty);
		
		int direction = ((int)((theta * HIST_DIRECTIONS)/360));
		pixel.x = direction * 360/HIST_DIRECTIONS;
		
		//float proportion = ((float)hist2d[direction][(int)(r*HIST_BINS)])/histsum2d[direction];
		
		pixel.y = r > UPPER2d[direction]*HIST_RESOLUTION/HIST_BINS ? 0 : 1;
		pixel.z = r > prop_above_upper[direction]*10 ? 0 : 1;
	});
	
	
	cvtColor(foo,foo,CV_HSV2BGR);
	imshow("Color Histogram",foo);
	
	return;

}

void wheel(){ //Display the color wheel
	
	namedWindow("Color Wheel", WINDOW_AUTOSIZE );
	
	Mat foo = Mat::ones(480, 480, CV_32FC3);
	
	foo.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
		
		float tx = (position[1]-240.0)/240.0;
		float ty = (position[0]-240.0)/240.0;
		
		float theta = atan2(ty,tx)*180/M_PI;//find angle
		theta += theta < 0 ? 360 : 0; //enforce strict positive angle
		float r = sqrt(tx*tx + ty*ty);
		
		int direction = ((int)((theta * HIST_DIRECTIONS)/360));
		pixel.x = direction * 360/HIST_DIRECTIONS;
		
		
		pixel.y = r > 1 ? 0 : 1;
		pixel.z = r > 1 ? 0 : 1;

		
	});
	
	
	cvtColor(foo,foo,CV_HSV2BGR);
	imshow("Color Wheel",foo);
	
	waitKey(0);
	
	exit(0);

	
}

void streamline_field(Pixel2 * pt, float* distancetraveled, int xoffset, int yoffset, cv::Mat flow, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]){
	
	
	for( int i = 0; i< iterations; i++){
		
		float x = pt->x + xoffset;
		float y = pt->y + yoffset;
		
		int xind = (int) floor(x);
		int yind = (int) floor(y);
		float xrem = x - xind;
		float yrem = y - yind;
		
		if(xind < 1 || yind < 1 || xind + 2 > flow.cols || yind  + 2 > flow.rows)  //Verify array bounds
		{
			return;
		}
		
		//Bilinear interpolation
		Pixel2 delta =		(*flow.ptr<Pixel2>(yind,xind))		* (1-xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind,xind+1))	* (xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind))	* (1-xrem)*(yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind+1))	* (xrem)*(yrem) ;
		
		
		float theta = atan2(delta.y,delta.x)*180/M_PI;//find angle
		theta += theta < 0 ? 360 : 0; //enforce strict positive angle
		int direction = ((int)((theta * HIST_DIRECTIONS)/360));
		if(prop_above_upper[direction] > .05){return;}
		
		float r = sqrt(delta.x*delta.x + delta.y*delta.y);
		if(r > UPPER){return;}
		
		Pixel2 newpt = *pt + delta*dt/iterations;
		*distancetraveled = *distancetraveled + r;
		
		*pt = newpt;
	}
	
	return;
}




void streamline(Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]){
	
	
	for( int i = 0; i< iterations; i++){
		
		float x = pt->x;
		float y = pt->y;
		
		int xind = (int) floor(x);
		int yind = (int) floor(y);
		float xrem = x - xind;
		float yrem = y - yind;
		
		if(xind < 1 || yind < 1 || xind + 2 > flow.cols || yind  + 2 > flow.rows)  //Verify array bounds
		{
			return;
		}
		
		//Bilinear interpolation
		Pixel2 delta =		(*flow.ptr<Pixel2>(yind,xind))		* (1-xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind,xind+1))	* (xrem)*(1-yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind))	* (1-xrem)*(yrem) +
		(*flow.ptr<Pixel2>(yind+1,xind+1))	* (xrem)*(yrem) ;
		
		
		float theta = atan2(delta.y,delta.x)*180/M_PI;//find angle
		theta += theta < 0 ? 360 : 0; //enforce strict positive angle
		int direction = ((int)((theta * HIST_DIRECTIONS)/360));
		if(prop_above_upper[direction] > .05){return;}
		
		float r = sqrt(delta.x*delta.x + delta.y*delta.y);
		if(r > UPPER){return;}
		
		
		Pixel2 newpt = *pt + delta*dt/iterations;
		
		cv::line(overlay,* pt, newpt, color, 1, 8, 0);
		
		*pt = newpt;
	}
	
	return;
}


