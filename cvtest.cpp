#include <math.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#define WIN_SIZE 30
#define STABILIZE 10

using namespace cv;

typedef cv::Point_<float> Pixel2;
typedef cv::Point3_<float> Pixel3;

void wheel();

int main(int argc, char** argv )
{
	
	
	//wheel();
	
	
	if(argc <2){printf("No video specified\n");wheel();}
	
	
	ocl::setUseOpenCL(true);
	
	VideoCapture video = VideoCapture(argv[1]);
	
	int c, r;
	c = (int) video.get(CAP_PROP_FRAME_WIDTH);
	r = (int) video.get(CAP_PROP_FRAME_HEIGHT);
	
	
	
	
	float scalex = 640.0/c;
	float scaley = 480.0/r;
	
	//printf("Dimensions: %d by %d, resized to %3.0f by %3.0f\n",r,c,r*scaley,c*scalex);
	
	
 
	Mat frame;
	Mat subframe[WIN_SIZE];
	Mat f2,f3;
	Mat flow_raw;
	
	Mat flow;
	Mat stable[STABILIZE];
	
	///float directions[10][65][49][36] = {0}; //This will be a massive array. Just yuuge. The biggest. Coordinates as follows: timeframe, column, row, direction bin. Value: intensity in the 20*20 sector in that direction.
	//printf("%d\n",sizeof(directions));
	//exit(0);
	
	Mat accumulator[WIN_SIZE]; //magnitude of some sort
	Mat accumulator2[WIN_SIZE]; //magnitude of some sort
	//switch to multiple views: high magnitude and low magnitude flow
	//A wave has high magnitude
	
	for(int j = 0 ; j< WIN_SIZE; j++){accumulator[j] = Mat::zeros(480, 640, CV_32FC1);}
	for(int j = 0 ; j< STABILIZE; j++){stable[j] = Mat::zeros(480, 640, CV_32FC2);}
	Mat ones = Mat::ones(480, 640, CV_32FC1);
	Mat out = Mat::zeros(480, 640, CV_32FC1);
	
	Mat splitarr[2];
	namedWindow("foo", WINDOW_AUTOSIZE );
	namedWindow("foo1", WINDOW_AUTOSIZE );
	
	
	int i;
	
	
	int turn = 0;
	video.read(frame);
	
	if(frame.empty()){exit(1);}
	resize(frame,subframe[0],Size(),scalex,scaley,INTER_AREA);
	cvtColor(subframe[0],f2,COLOR_BGR2GRAY);
	
	
	
	//exit(0);
	int hist[100] = {0};

	
	for( i = 1; true; i++){//fix video read, right now it's 1/4 realtime.

		
		
		video.read(frame);
		if(frame.empty()){break;}
		resize(frame,subframe[i%WIN_SIZE],Size(),scalex,scaley,INTER_AREA);
		
		if(turn){
			cvtColor(subframe[i%WIN_SIZE],f2,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f3,f2, flow_raw, 0.5, 3, 5, 3, 15, 1.2, 0);
			//printf("tick\n");
		}else{
			cvtColor(subframe[i%WIN_SIZE],f3,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f2,f3, flow_raw, 0.5, 3, 5, 3, 15, 1.2, 0);
			//printf("tock\n");
		}
		turn = !turn;
		
		for(int j = 0; j<STABILIZE; j++){
			add(flow_raw,stable[j],stable[j]);
		}
	
		//Scalar fmean,fstddev;
		//meanStdDev(flow_raw,fmean,fstddev,noArray());
		
		
		Mat current = stable[i%STABILIZE];
		
		current.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {
			
			float tx = pixel.x;
			float ty = -pixel.y;
			
			
			int bin = (int) floor(atan(ty/tx)/M_PI  *18  );//Begins to calculate the angle as an integer between 1 and 35.
			
			
			
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
			pixel.x = (float)bin*10;
			
			
			
			pixel.y = sqrt(tx*tx + ty*ty);
			
		});
		
		
		
		
		float meanflow = 0;
		int count = 0;
		
		float maxflow = 0;
		
		
		for (int y = 0; y < 480; y++) {
			Pixel2* ptr = current.ptr<Pixel2>(y, 0);
			const Pixel2* ptr_end = ptr + 640;
			for (int x = 0 ; ptr != ptr_end; ++ptr, x++) {
				{meanflow += ptr->y;count++;hist[(int)floor(ptr->y/5)]++;}
			//	if(ptr->y > maxflow){maxflow = ptr->y;}
			}
		}

		
		
		meanflow /= count; //need to use statistical methods to find mean instead
		
		//printf("%f\n",meanflow);
		//flow_raw.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void { if(pixel.x < 180 && pixel.x > 0 && pixel.y > meanflow ){pixel.y /= meanflow;}else{pixel.y = 0;}});
		current.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void {pixel.y /= STABILIZE;  if(pixel.y > 1 || pixel.y < 0  /*|| pixel.x < 0 || pixel.x >180*/){pixel.y = 0;}else{}});
		//flow_raw.forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void { pixel.y /= 10;});
		split(current,splitarr);

		
		
		flow = Mat(480, 640, CV_32FC3);
		Mat conv[] = {splitarr[0],splitarr[1],splitarr[1]};
		merge(conv,3,flow);
		
		for(int j = 0; j<WIN_SIZE; j++){
			add(splitarr[1],accumulator[j],accumulator[j]);
		}
		
		multiply(accumulator[i%WIN_SIZE],ones,out,1.0);
		
		cvtColor(flow,flow,CV_HSV2BGR);
		
		
		imshow("foo",flow);
		imshow("foo1",subframe[i%WIN_SIZE]);
		
		accumulator[i%WIN_SIZE] = Mat::zeros(480, 640, CV_32FC1);
		stable[i%STABILIZE] = Mat::zeros(480, 640, CV_32FC2);
		
		waitKey(1);
		
	}
	/*
	for(int i = 0; i < 40; i++){
		printf("%5d ",hist[i]);
	}
	*/
	printf("\n");
	return 0;
	
}


void wheel(){
	
	namedWindow("Color Wheel", WINDOW_AUTOSIZE );
	
	Mat foo = Mat::ones(480, 480, CV_32FC3);
	
	foo.forEach<Pixel3>([&](Pixel3& pixel, const int position[]) -> void {
		
		float tx = (position[1]-240)/240.0;
		float ty = (240-position[0])/240.0;
		
		
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
	
	cvtColor(foo,foo,CV_HSV2BGR);
	imshow("Color Wheel",foo);
	waitKey(0);
	
	exit(0);

	
}

