#include <math.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;

typedef cv::Point3_<float> Pixel;

int main(int argc, char** argv )
{
	
	if(argc <2){printf("No video specified\n");exit(0);}

	
	ocl::setUseOpenCL(true);
	
	VideoCapture video = VideoCapture(argv[1]);

	int c, r;
	c = (int) video.get(CAP_PROP_FRAME_WIDTH);
	r = (int) video.get(CAP_PROP_FRAME_HEIGHT);



	
	float scalex = 640.0/c;
	float scaley = 480.0/r;
	
	printf("Dimensions: %d by %d, resized to %3.0f by %3.0f\n",r,c,r*scaley,c*scalex);
	
	

	Mat frame;
	Mat subframe[90];
	Mat f2,f3;
	Mat flow_raw;

	Mat flow[90];
	
	float directions[10][65][49][36] = {0}; //This will be a massive array. Just yuuge. The biggest. Coordinates as follows: timeframe, column, row, direction bin. Value: intensity in the 20*20 sector in that direction.
	//printf("%d\n",sizeof(directions));
	//exit(0);
	Mat splitarr[2];
	namedWindow("foo", WINDOW_AUTOSIZE );
	

	int i;


	int turn = 0;	
	video.read(frame);
	
	if(frame.empty()){exit(1);}
	resize(frame,subframe[0],Size(),scalex,scaley,INTER_AREA);
	cvtColor(subframe[0],f2,COLOR_BGR2GRAY);

	
	
	//exit(0);
	
	
	for( i = 1; i < 90; i++){//fix video read, right now it's 1/4 realtime.
		int timeslot = i / 90;
		
		video.read(frame);
		if(frame.empty()){break;}
		resize(frame,subframe[i],Size(),scalex,scaley,INTER_AREA);
		
		if(turn){
			cvtColor(subframe[i],f2,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f3,f2, flow_raw, 0.5, 2, 5, 2, 3, 1.2, 0);
			//printf("tick\n");
		}else{
			cvtColor(subframe[i],f3,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f2,f3, flow_raw, 0.5, 2, 5, 2, 3, 1.2, 0);
			//printf("tock\n");
		}
		turn = !turn;
	
		Scalar fmean,fstddev;
		
		meanStdDev(flow_raw,fmean,fstddev,noArray());
		
		float normalize = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]) + sqrt(fstddev[0] * fstddev[0] + fstddev[1] * fstddev[1]);
		float meanflow = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]);
		
		//Scalar  fmean = mean(flow);
		//float meanval = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]);
		//printf("%f\n",meanval);
		split(flow_raw,splitarr);
		
		
		
		Mat blank = Mat::zeros(Size(flow_raw.cols, flow_raw.rows), CV_32FC1);
		flow[i-1] = Mat(flow_raw.rows, flow_raw.cols, CV_32FC3);
		Mat conv[] = {splitarr[0],splitarr[1],blank};
		merge(conv,3,flow[i-1]);

		int bins[36] = {0};

		
		flow[i-1].forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
			float tx = pixel.x;
			float ty = pixel.y;
			
			int bin = (int) floor(atan(ty/tx)/M_PI  *36  );//Begins to calculate the angle as an integer between 1 and 35.
			
			
			
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
			
			float mag = sqrt(tx*tx + ty*ty);
			
			int binx = ceilf(position[0]/10);
			int biny = ceilf(position[1]/10);
			if(bin > -1 && bin < 36 && isfinite(mag)){ //increment a counter.
				directions[timeslot][binx][biny][bin]+=mag;//as long as the direction and magnitude are real, increment the four counters
				directions[timeslot][binx+1][biny][bin]+=mag;
				directions[timeslot][binx][biny+1][bin]+=mag;
				directions[timeslot][binx+1][biny+1][bin]+=mag;
			}
			//There is a rotating queue of bins. Implement a 90 frame queue, perhaps?
			//90 frame main queue 640*480 3channel (circular).
			//90 (or some divisor thereof) queue of bins 64*48? 100*100?  36 channel. Overlap must be planned.
			/*
			 20*20 subsections. Spaced by 10 (each point except for the edges is in 4 regions. A total of (64 -1)*(48 -1) squares? 64*48, discard outer edge
			 */
			
			pixel.x = (float)bin*10.0; //Turns previous angle into a standard 360 degree representation.
			
			
			if(mag > meanflow){ mag = meanflow/normalize;} else{ mag = 0;}
			pixel.y = mag;
			pixel.z = mag;
			
		});

		
		
		cvtColor(flow[i-1],flow[i-1],CV_HSV2BGR);
		
		imshow("foo",flow[i-1]);
		
		waitKey(1);
	
	}
	
	for(int y = 1; y < 48; y++){
		for(int x = 1; x < 64; x++){
			float maxval = 1;
			int maxbin = -1;
			for(int i = 0; i < 36; i++){
				if(directions[0][x][y][i]>maxval){maxval = directions[0][x][y][i]; maxbin = i;}
			}printf("%2d ",maxbin);
			
		}
		printf("\n");
		
	}

    return 0;

}
