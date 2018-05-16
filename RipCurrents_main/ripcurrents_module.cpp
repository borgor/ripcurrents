#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name

#include "ripcurrents.hpp"

// Mat streamfield		-input: how far it moved
// Mat streamoverlay_color		-output color
void streamline_displacement(Mat& streamfield, Mat& streamoverlay_color){
	//Compute streamline length
	double lenmax;
	minMaxLoc(streamfield,NULL,&lenmax,NULL,NULL);
	(streamfield).convertTo(streamoverlay_color,CV_8UC1,255/lenmax);
	applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
}

// Mat streamlines_distance		-input: how far it has moved
// Mat streamoverlay_color		-output color
void streamline_total_motion(Mat& streamlines_distance, Mat& streamoverlay_color){
	double distmax;
	minMaxLoc(streamlines_distance,NULL,&distmax,NULL,NULL);
	streamlines_distance.convertTo(streamoverlay_color,CV_8UC1,255/distmax);
	applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
}

// Mat streamfield		-input: how far it moved
// Mat streamlines_distance		-input: how far it has moved
// Mat streamoverlay_color		-output color
// pre: streamline_displacement(), streamline_total_motion()
void streamline_ratio(Mat& streamfield, Mat& streamlines_distance, Mat& streamoverlay_color){
	divide(streamfield,streamlines_distance,streamoverlay_color);
	double ratiomax;
	minMaxLoc(streamoverlay_color,NULL,&ratiomax,NULL,NULL);
	streamoverlay_color.convertTo(streamoverlay_color,CV_8UC1,255/ratiomax);
	applyColorMap(streamoverlay_color, streamoverlay_color, COLORMAP_JET);
}

// Mat streamlines_mat		-input
// Mat streamline_density		-output
void streamline_positions(Mat& streamlines_mat, Mat& streamline_density){
	for (int y = 0; y < YDIM; y++) {
		Pixel2* ptr = streamlines_mat.ptr<Pixel2>(y, 0);
		const Pixel2* ptr_end = ptr + (int)XDIM;
		for (int x = 0 ; ptr != ptr_end; ++ptr, x++) {
			int xind = (int) roundf(floor(ptr->x + x));
			int yind = (int) roundf(floor(ptr->y + y));
			if(xind < 1 || yind < 1 || xind + 2 > streamline_density.cols || yind  + 2 > streamline_density.rows)  //Verify array bounds
				{continue;}

			Pixel3 * density_ptr = streamline_density.ptr<Pixel3>(yind,xind);
			(*density_ptr) = Pixel3(1,1,1);
			
		}
	}
}

// Mat streamout		-output
// Mat streamoverlay_color
// Mat streamoverlay
// int streamlines		-number of streamline track point
// Pixel2 streampt[]	-array of streamline track point
// int framecount
// int totalframes
// Mat current
// float UPPER
// float prop_above_upper
void get_streamlines(Mat& streamout, Mat& streamoverlay_color, Mat& streamoverlay, int streamlines, Pixel2 streampt[], int framecount, int totalframes, Mat& current, float UPPER, float prop_above_upper[]){
	for(int s = 0; s < streamlines; s++){
		streamline(streampt+s, Scalar(framecount*(255.0/totalframes)), current, streamoverlay, 2, 1,UPPER,prop_above_upper);
	}
	
	applyColorMap(streamoverlay, streamoverlay_color, COLORMAP_RAINBOW);
	add(streamoverlay_color, streamout, streamout, streamoverlay, -1);
}

// Mat current
// int hist[]		-
// int histsum		-
// int hist2d[][]		-
// int histsum2d[]		-
// float UPPER		-
// float UPPER2d[]		-
// float prop_above_upper[]		-
void create_histogram(Mat current, int hist[HIST_BINS], int& histsum, int hist2d[HIST_DIRECTIONS][HIST_BINS]
	 ,int histsum2d[HIST_DIRECTIONS], float& UPPER, float UPPER2d[HIST_DIRECTIONS], float prop_above_upper[HIST_DIRECTIONS]){
	
	//Construct histograms to get thresholds
	//Figure out what "slow" or "fast" is
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
	//This is more of a visual aid, allowing small motion in directions with 
	//little movement to not be drowned out by large-scale motion
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
}

// Mat current
// Mat waterclass
// Mat accumulator2
// UPPER
// MID
// LOWER
// UPPER2d[]
void create_flow(Mat current, Mat waterclass, Mat accumulator2, float UPPER, float MID, float LOWER, float UPPER2d[HIST_DIRECTIONS]){
	//Classify into fast and slow movement
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
}

// Mat accumulator
// Mat accumulator2
// Mat out
// Mat outmask
// int framecount
void create_accumulationbuffer(Mat& accumulator, Mat accumulator2, Mat& out, Mat outmask, int framecount){
//For every pixel, if it has been identified as fast(possibly a wave), increment a counter.
	if(framecount > 30){
		add(accumulator2,accumulator,accumulator);
	}
	
	//Visualize the accumulated waves
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
}

// Mat outmask
// pre: accumulationbuffer()
void create_edges(Mat& outmask){
	Mat morph_window = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
	dilate(outmask, outmask, morph_window);
	morphologyEx( outmask, outmask, 4, morph_window ); //Morphology operation #4: gradient (dilation - erosion)	
}

// Mat subframe		:output
// Mat outmask		:input
// pre: accumulationbuffer(), create_edges()
void create_output(Mat& subframe, Mat outmask){
	Mat overlay = Mat::zeros(Size(XDIM, YDIM), CV_8UC3);
	
	//Combine edges and original
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
		//if(prop_above_upper[direction] > .05){return;}
		
		float r = sqrt(delta.x*delta.x + delta.y*delta.y);
		if(r > UPPER){return;}
		
		
		Pixel2 newpt = *pt + delta*dt/iterations;
		
		cv::line(overlay,* pt, newpt, color, 1, 8, 0);
		
		*pt = newpt;
	}
	
	return;
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
		//if(prop_above_upper[direction] > .05){return;}
		
		float r = sqrt(delta.x*delta.x + delta.y*delta.y);
		if(r > UPPER){return;}
		
		Pixel2 newpt = *pt + delta*dt/iterations;
		*distancetraveled = *distancetraveled + r;
		
		*pt = newpt;
	}
	
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