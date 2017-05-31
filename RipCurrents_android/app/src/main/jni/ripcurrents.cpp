#include <math.h>
#include <stdio.h>
#include <android/log.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name

#include "ripcurrents.h"

#define STABILIZE 2	//Size of buffer for stabilizing video
#define THRESH_BINS 100 //Number of bins for finding thresholds


#define DEBUG_TAG "gDebug"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, DEBUG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, DEBUG_TAG, __VA_ARGS__)

//Some thresholds to mask out any remaining jitter, and strong waves. Don't know how to calculate them.
float LOWER =  0.2;
float MID  = .5;
float UPPER = 100.0; //UPPER can be determined programmatically

#define rescale(x) x = (x - LOWER)/(UPPER - LOWER)

using namespace cv;

typedef cv::Point3_<char> Pixelc;
typedef cv::Point_<float> Pixel2;
typedef cv::Point3_<float> Pixel3;

void wheel(){
    Mat frame(480, 480, CV_32FC3);

        frame.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
            float tx = (position[1]-480/2)/480.0;
            float ty = (position[0]-480/2)/480.0;
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
        });
        cvtColor(frame,frame,CV_HSV2BGR);
        frame.convertTo(frame,CV_8UC3,255);

        std::vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);

        imwrite("/storage/emulated/0/wheel.png",frame,compression_params);
        LOGD("Generated wheel");
        return;
}

double timediff(){
	static struct timeval oldtime;
	struct timeval newtime;
	gettimeofday(&newtime,NULL);
	double diff = (newtime.tv_sec - oldtime.tv_sec) + (newtime.tv_usec - oldtime.tv_usec)/1000000.0;
	gettimeofday(&oldtime,NULL);
	return diff;
}

int rip_main(cv::VideoCapture video, cv::VideoWriter video_out){
	double time_farneback = 0;
	double time_polar = 0;
	double time_threshold = 0;
	double time_overlay = 0;
	int frames_read = 0;
	timediff();

	int c, r;
	c = (int) video.get(CAP_PROP_FRAME_WIDTH);
	r = (int) video.get(CAP_PROP_FRAME_HEIGHT);


	float scalex = XDIM/c;
	float scaley = YDIM/r;

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



	int hist[THRESH_BINS] = {0}; //histogram
	int histsum = 0;



	int i; //Generic iterator for main loop.


	int turn = 0;  //Alternator

	//Preload a frame
	video.read(frame);
	if(frame.empty()){exit(1);}
	resize(frame,subframe,Size(),scalex,scaley,INTER_AREA);
	cvtColor(subframe,f2,COLOR_BGR2GRAY);
	f2.copyTo(u_f1);





	for( i = 1; true; i++){



		video.read(frame);
		video.read(frame); //skip frames for speed
		video.read(frame);
		frames_read+=3;
		LOGD("Frames read: %d\n",frames_read);

		if(frame.empty()){break;}
		timediff();


		//Resize, turn to gray.
		resize(frame,subframe,Size(),scalex,scaley,INTER_AREA);
		cvtColor(subframe,f2,COLOR_BGR2GRAY);
		if(turn){
			f2.copyTo(u_f1);
			calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 3, 5, 3, 15, 1.2, 0); //Give to GPU
			//printf("tick\n");
		}else{
			f2.copyTo(u_f2);
			calcOpticalFlowFarneback(u_f1,u_f2, u_flow, 0.5, 3, 5, 3, 15, 1.2, 0);
			//printf("tock\n");
		}
		turn = !turn;

		flow_raw = u_flow.getMat(ACCESS_READ); //Tell GPU to give it back

		for(int j = 0; j<STABILIZE; j++){
			add(flow_raw,stable[j],stable[j]);
		}

		Mat current = stable[i%STABILIZE]*(1.0/STABILIZE);

		time_farneback += timediff();

		split(current,splitarr);
		cartToPolar(splitarr[0], splitarr[1], splitarr[1], splitarr[0],true);
		merge(splitarr,2,current);


		time_polar += timediff();


		int resolution = 10;
		//Fill histogram
		for (int y = 0; y < YDIM; y++) {
			Pixel2* ptr = current.ptr<Pixel2>(y, 0);
			const Pixel2* ptr_end = ptr + (int)XDIM;
			for (int x = 0 ; ptr != ptr_end; ++ptr, x++) {
				int bin = (ptr->y) * resolution;
				if(bin < THRESH_BINS &&  bin >= 0) {hist[bin]++; histsum++;}

			}
		}

		//Use histogram to create threshold
		int threshsum = 0;
		int bin = THRESH_BINS-1;
		while(threshsum < (histsum*.03)){
			threshsum += hist[bin];
			bin--;
		}
		UPPER = bin/float(resolution);
		//printf("%f\n",UPPER);



		Mat accumulator2 = Mat::zeros(YDIM, XDIM, CV_32FC3);
		Mat waterclass = Mat::zeros(YDIM, XDIM, CV_32FC3);


		//Classify
		current.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {

			Pixel3* classptr = waterclass.ptr<Pixel3>(position[0],position[1]);

			Pixel3 * pt = accumulator2.ptr<Pixel3>(position[0],position[1]);
			float dir = pixel.x;
			float val = pixel.y ;
			if(val > UPPER){classptr->x = .5; pt->x++;}else{
				if(val > MID){classptr->z = 1;}else{
					if(val > LOWER){classptr->z = .5;}else{
						 {classptr->y = .5;}
					}
				}
			}

			if(val < UPPER){
				if(val> LOWER){
					rescale(val);
					pixel.y = val;
				}else{
					pixel.y = 0;
				}
			}
		});



		/*
		//Convert flow to HSV, then BGR, then display
		split(current,splitarr);
		flow = Mat(YDIM, XDIM, CV_32FC3);
		Mat conv[] = {splitarr[0],splitarr[1],splitarr[1]};
		merge(conv,3,flow);
		cvtColor(flow,flow,CV_HSV2BGR);
		flow.convertTo(flow,CV_8UC3,255);
		//imshow("Flow",flow);
		*/

		time_threshold += timediff();

		//accumulator accumulates waves
		add(accumulator2,accumulator,accumulator);



		Mat out = Mat::zeros(YDIM, XDIM, CV_32FC3);

		//Identify rip currents
		out.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
			Pixel3* accptr = accumulator.ptr<Pixel3>(position[0],position[1]);

			int val = accptr->x;
			if(val > .05 * i){
				if(val < .2 * i){
					pixel.z = float(val) / i;
				}else{
					pixel.x = float(val) / i;
				}
			}else{
					pixel.y = float(val) / i;
			}
		});

		Mat overlay = Mat::zeros(YDIM, XDIM, CV_8UC3);

		//Find green surrounded by red in accumulator image, create overlay
#define localwin 20

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


		//Combine overlay and original
		if(i>90){
			subframe.forEach<Pixelc>([&](Pixelc& pixel, const int position[]) -> void {
				Pixelc* over = overlay.ptr<Pixelc>(position[0],position[1]);
				if(over->z == 4){
					pixel.z = 255;
				}
			});
		}

		time_overlay += timediff();

		video_out.write(subframe);


		stable[i%STABILIZE] = Mat::zeros(YDIM, XDIM, CV_32FC2);



	}
	LOGD("Frames read: %d\n",frames_read);
	LOGD("Time spent on farneback: %f\n",time_farneback);
	LOGD("Time spent on polar coordinates: %f\n",time_polar);
	LOGD("Time spent on thresholds: %f\n",time_threshold);
	LOGD("Time spent on overlay: %f\n",time_overlay);

	//Clean up

	flow_raw.release();

	//waitKey(0);
	video.release();
	video_out.release();

	return 0;
}