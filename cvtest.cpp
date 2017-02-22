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
	
	Mat frame,f1,f2,f3,flow,flow2;
	Mat splitarr[2];
	namedWindow("foo", WINDOW_AUTOSIZE );
	

	int i;


	int turn = 0;	
	video.read(frame);
	
	if(frame.empty()){exit(1);}
	resize(frame,f1,Size(),scalex,scaley,INTER_AREA);
	cvtColor(f1,f2,COLOR_BGR2GRAY);


	for( i = 0; true; i++){//fix video read, right now it's 1/4 realtime.
		
		
		video.read(frame);
		if(frame.empty()){break;}
		resize(frame,f1,Size(),scalex,scaley,INTER_AREA);
		
		if(turn){
			cvtColor(f1,f2,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f3,f2, flow, 0.5, 2, 5, 2, 3, 1.2, 0);
			//printf("tick\n");
		}else{
			cvtColor(f1,f3,COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(f2,f3, flow, 0.5, 2, 5, 2, 3, 1.2, 0);
			//printf("tock\n");
		}
		turn = !turn;
	
		Scalar fmean,fstddev;
		
		meanStdDev(flow,fmean,fstddev,noArray());
		
		float normalize = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]) + sqrt(fstddev[0] * fstddev[0] + fstddev[1] * fstddev[1]);
		float meanflow = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]);
		
		//Scalar  fmean = mean(flow);
		//float meanval = sqrt(fmean[0] * fmean[0] + fmean[1] * fmean[1]);
		//printf("%f\n",meanval);
		split(flow,splitarr);
		
		
		
		Mat blank = Mat::zeros(Size(flow.cols, flow.rows), CV_32FC1);
		Mat flow3(flow.rows, flow.cols, CV_32FC3);
		Mat conv[] = {splitarr[0],splitarr[1],blank};
		merge(conv,3,flow3);

		int bins[36] = {0};

		
		flow3.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
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
			
			if(bin > -1 && bin < 36){ //increment a counter. Bins needs to be replaced by a metric ton of multidimensional arrays
				bins[bin]++;
			}
			//There is a rotating queue of bins. Implement a 90 frame queue, perhaps?
			//90 frame main queue 640*480 3channel (circular).
			//90 (or some divisor thereof) queue of bins 64*48? 100*100?  36 channel. Overlap must be planned.
			/*
			 20*20 subsections. Spaced by 10 (each point except for the edges is in 4 regions. A total of (64 -1)*(48 -1) squares? 64*48, discard outer edge
			 */
			
			pixel.x = (float)bin*10.0; //Turns previous angle into a standard 360 degree representation.
			
			float mag = sqrt(tx*tx + ty*ty);
			if(mag > meanflow){ mag = meanflow/normalize;} else{ mag = 0;}
			pixel.y = mag;
			pixel.z = mag;
			
		});

		
		int maxval = 0;
		int maxbin = 0;
		
		for(int i = 0; i < 36; i++){
			if(bins[i]>maxval){maxval = bins[i]; maxbin = i;}
		}printf("%d\n",maxbin);
		
		cvtColor(flow3,flow3,CV_HSV2BGR);
		
		imshow("foo",flow3);
		
		waitKey(1);
	
	}
	

    return 0;

}
