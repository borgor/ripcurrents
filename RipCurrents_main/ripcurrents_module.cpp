#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name
#include <opencv2/optflow/motempl.hpp>

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
		printf("%f\n", streampt[s].x);
		streamline(streampt+s, Scalar(framecount*(255.0/totalframes)), current, streamoverlay, 0.1, 100, UPPER,prop_above_upper);
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
		//const Pixel3* ptr_end = ptr + (int)XDIM;
		const Pixel3* ptr_end = current.ptr<Pixel3>(y+1, 0);
		for (; ptr < ptr_end; ++ptr) {
			int bin = (ptr->y) * HIST_RESOLUTION;
			int angle = (ptr->x * HIST_DIRECTIONS)/ 360; //order matters, truncation
			if(bin < HIST_BINS &&  bin >= 0){
				hist[bin]++; histsum++;
				hist2d[angle][bin]++;
				histsum2d[angle]++;
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

void stabilizer(Mat current, Mat current_prev){
	double sum_x=0, sum_y=0;
	int cx=0, cy=0;
	int diff = 0;
	for ( int row = (int)(current.rows * 0.9); row < current.rows; row++ ){
		Pixel2* ptr = current.ptr<Pixel2>(row, (int)(current.cols * 0.9));
		cy++;
		cx=0;
		for ( int col = (int)(current.cols * 0.9); col < current.cols; col++){
			sum_x += ptr->x;
			sum_y += ptr->y;
			cx++;
			ptr++;
		}
	}

	double mean_x = sum_x / cx;
	double mean_y = sum_y / cy;

	printf("x %f, y %f \n", mean_x, mean_y);
	
	for ( int row = 0; row < current.rows; row++ ){
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < current.cols; col++){
			if(ptr->x != 0) ptr->x -= mean_x * 0.2;
			if(ptr->y != 0) ptr->y -= mean_y * 0.2;
			ptr++;
		}
	}
}


/**
 * @fn
 * Find the average vector of every pixel with motempl::calcGlobalOrientation
 * @brief Find the global orientation
 * @param (UMat u_f1) Resized and gray-scaled previous frame
 * @param (UMat u_f2) Resized and gray-scaled current frame
 * @param (Mat& hist_gray) Return the image of the global orientation vector on the current frame image
 */
void globalOrientation(UMat u_f1, UMat u_f2, Mat& hist_gray){
	Mat color_diff;
	absdiff(u_f1, u_f2, color_diff);
	Mat black_diff;
	threshold(color_diff, black_diff, 30, 1, THRESH_BINARY);
	clock_t proc_time = clock();
	Mat motion_history = Mat::zeros(Size(XDIM, YDIM), CV_32FC1);
	motempl::updateMotionHistory(black_diff, motion_history, proc_time, 1);
	// min max normalize
	double max, min;
	minMaxLoc(motion_history, &min, &max);
	motion_history = motion_history / max;
	Mat mask, orientation;
	motempl::calcMotionGradient(motion_history, mask, orientation, 0.25, 1, 3);
	double angle_deg = motempl::calcGlobalOrientation(orientation, mask, motion_history, proc_time, 1);
	printf("%lf\n", angle_deg);
	Mat hist_color = (motion_history ) * 255;
	hist_color.convertTo(hist_color, CV_8U);
	cvtColor(hist_color,hist_gray,COLOR_GRAY2BGR);

	// draw center point
	circle(hist_gray, Point((int)(XDIM / 2), (int)(YDIM / 2)), 3, Scalar(0, 215, 255), CV_FILLED, 16, 0);

	// draw line
	double angle_rad = angle_deg * M_PI / 180;
	arrowedLine(hist_gray, Point((int)(XDIM / 2), (int)(YDIM / 2)), 
		Point((int)(XDIM / 2 + cos(angle_rad) * 10), (int)(YDIM / 2 + sin(angle_rad) * 50)),
		Scalar(0, 215, 255), 2, 16, 0, 0.2);

	// draw dots
	for ( int row = 0; row < YDIM; row += 30 ){
		for ( int col = 0; col < XDIM; col += 30 ){
			circle(hist_gray, Point(col, row), 1, Scalar(0, 215, 0), CV_FILLED, 16, 0);
			angle_deg = orientation.at<double>(row, col);
			if ( angle_deg > 0 ) angle_rad = angle_deg * M_PI / 180;
			arrowedLine(hist_gray, Point(col, row), 
				Point((int)(col + cos(angle_rad) * 10), (int)(row + sin(angle_rad) * 10)),
				Scalar(0, 215, 0), 1, 16, 0, 0.4);
		}
	}
}

// subframe - bgr image of current frame
// buffer_hsv - store hsv format data of previous BUFFER_COUNT frames
// update_ith_buffer - number of element in buffer array to update
// average_hsv - average hsv of buffer_hsv
void averageHSV(Mat& subframe, std::vector<Mat> buffer_hsv, int update_ith_buffer, Mat& average_hsv){
	Mat hsv;
	//cvtColor(subframe, hsv, COLOR_HSV2BGR);

	// subtract old buffer data from average
	average_hsv -= buffer_hsv[update_ith_buffer];

	// get new buffer
	buffer_hsv[update_ith_buffer] = subframe / BUFFER_FRAME;
	// add new buffer to average
	average_hsv += buffer_hsv[update_ith_buffer];
}

// buffer - store previous BUFFER_COUNT frames
// current - frame data
// update_ith_buffer - number of element in buffer array to update
// average - store the average vector data
// average_color - convert the vector data to hsv format image
// grid - average of average in small grid
// max_displacement - store the max displacement of vector
// UPPER - histogram data to get clear result
void averageVector(std::vector<Mat> buffer, Mat& current, int update_ith_buffer, Mat& average, Mat& average_color, double** grid, float max_displacement, float UPPER) {
	// number of rows and cols in each grid
	int grid_col_num = (int)(XDIM/GRID_COUNT);
	int grid_row_num = (int)(YDIM/GRID_COUNT);
	
	// subtract old buffer data from average
	average -= buffer[update_ith_buffer] / BUFFER_FRAME;
	buffer[update_ith_buffer] = Mat::zeros(YDIM,XDIM,CV_32FC2);
	// get new buffer
	buffer[update_ith_buffer].forEach<Pixel2>([&](Pixel2& pixel, const int position[]) -> void{
		get_delta(&pixel, position[1],position[0], current, 2, UPPER);
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
			//if ( ptr2->z < 30 ) ptr2->z = 0;

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
			double angle_between = min(abs(angle_rad - global_angle_rad), 2*M_PI-abs(angle_rad - global_angle_rad));
			if ( angle_between > M_PI * 0.7 ) {
				circle(average_color, Point(col * grid_col_num, row * grid_row_num), 1, Scalar(0, 255, 0), CV_FILLED, 16, 0);
				arrowedLine(average_color, Point(col * grid_col_num, row * grid_row_num), 
					Point((int)(col * grid_col_num + cos(angle_rad) * 10), (int)(row * grid_row_num + sin(angle_rad) * 10)),
					Scalar(0, 255, 0), 1, 16, 0, 0.4);
			} /*else {
				circle(average_color, Point(col * grid_col_num, row * grid_row_num), 1, Scalar(255, 0, 0), CV_FILLED, 16, 0);
				arrowedLine(average_color, Point(col * grid_col_num, row * grid_row_num), 
					Point((int)(col * grid_col_num + cos(angle_rad) * 10), (int)(row * grid_row_num + sin(angle_rad) * 10)),
					Scalar(255, 0, 0), 1, 16, 0, 0.4);
			}*/
		}
	}
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
		
		
		Pixel2 newpt = *pt + delta*dt;
		
		cv::line(overlay,* pt, newpt, color, 1, 8, 0);
		
		*pt = newpt;
	}
	
	return;
}


void streamline_2(Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]){
	
	
	for( int i = 0; i < iterations; i++){
		
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
		
		float r = sqrt(delta.x*delta.x + delta.y*delta.y);
		if(r > 5){return;}

		// printf("x: %f y: %f\n", delta.x, delta.y);
		
		
		Pixel2 newpt = *pt + delta * dt;
		
		cv::line(overlay,* pt, newpt, color, 1, 8, 0);
		
		*pt = newpt;
	}
	
	return;
}


void streamline_3(Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations, float UPPER, float prop_above_upper[HIST_DIRECTIONS]){
	
	
	for( int i = 0; i< 100; i++){
		
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
		
		
		
		Pixel2 newpt = *pt + delta*0.1;
		
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

void get_delta(Pixel2 * pt, int xoffset, int yoffset, cv::Mat flow, float dt, float UPPER){
	
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
	
	float r = sqrt(delta.x*delta.x + delta.y*delta.y);
	if(r > UPPER){return;}
	
	Pixel2 newpt = *pt + delta*dt;
	
	*pt = newpt;
	
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

/**
 * @fn
 * Compare the run time of two types of optical flow algorithm: calcOpticalFlowFarneback and calcOpticalFlowPyrLK
 * @param (UMat u_f1) UMat converted previous frame image
 * @param (UMat u_f2) UMat converted current frame image
 */
void farnebackAndLkSpeedComparison ( UMat u_f1, UMat u_f2 ) {
	UMat u_flow;

	// track points for calcOpticalFlowPyrLK
	std::vector<Point2f> features_prev, features_next;
	for ( int y = 0; y < YDIM; y++ ) {
		for ( int x = 0; x < XDIM; x++ ) {
			features_prev.push_back(Point2f(x, y));
		}
	}
	// return status values of calcOpticalFlowPyrLK
	std::vector<uchar> status;
	std::vector<float> err;

	// Run Farneback
	clock_t farne_start = clock();
	calcOpticalFlowFarneback(u_f2,u_f1, u_flow, 0.5, 2, 3, 2, 15, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN); //Give to GPU, possibly
	clock_t farne_end = clock();

	clock_t lk_start = clock();
	calcOpticalFlowPyrLK(u_f1, u_f2, features_prev, features_next, status, err, Size(21,21), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 1e-4 );
	clock_t lk_end = clock();

	std::cout << "farne back " << farne_end - farne_start << "\n";
	std::cout << "lk " << lk_end - lk_start << "\n";
}

/**
 * @fn
 * Let generated seed points flow
 * @param (UMat u_f1) UMat converted previous frame image
 * @param (UMat u_f2) UMat converted current frame image
 * @param (Mat subframe) current frame raw image
 * @param (std::vector<Point2f>& features_prev) input seed points
 * @param (std::vector<Point2f>& features_next) output seed points
 */
void flowRedPoints ( UMat u_f1, UMat u_f2, Mat subframe, std::vector<Point2f>& features_prev, std::vector<Point2f>& features_next ) {

	// return status values of calcOpticalFlowPyrLK
	std::vector<uchar> status;
	std::vector<float> err;
	
	calcOpticalFlowPyrLK(u_f1, u_f2, features_prev, features_next, status, err, Size(21,21), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1), 0, 1e-4 );
	features_prev = features_next;

	Mat features;
	subframe.copyTo(features);

	for ( int i = 0; i < (int)features_next.size(); i++ ) {
		circle(features,cvPoint(features_next[i].x,features_next[i].y),2,CV_RGB(100,0,0),-1,8,0);
	}

	imshow("features", features);
}

Timeline::Timeline(Pixel2 lineStart, Pixel2 lineEnd, int numberOfVertices){
	
	// define the distance between each vertices
	float diffX = (lineEnd.x - lineStart.x) / numberOfVertices;
	float diffY = (lineEnd.y - lineStart.y) / numberOfVertices;

	// create and push Pixel2 points
	for (int i = 0; i <= numberOfVertices; i++)
	{
		vertices.push_back(Pixel2(lineStart.x + diffX * i, lineStart.y + diffY * i));
	}

}

void Timeline::runLK(UMat u_prev, UMat u_current, Mat& outImg) {

	// return status values of calcOpticalFlowPyrLK
	vector<uchar> status;
	vector<float> err;

	// output locations of vertices
	vector<Pixel2> vertices_next;

	// run LK for all vertices
	calcOpticalFlowPyrLK(u_prev, u_current, vertices, vertices_next, status, err, Size(50,50),3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1), 10, 1e-4 );

	/*
	// eliminate any large movement
	for ( int i = 0; i < (int)vertices_next.size(); i++) {
		if ( abs(vertices.at(i).x - vertices_next.at(i).x) > XDIM * 0.1 
			|| abs(vertices.at(i).y - vertices_next.at(i).y) > YDIM * 0.1 ) {
				vertices_next.at(i) = vertices.at(i);
			}
	}
	*/

	// copy the result for the next frame
	vertices = vertices_next;

	/*
	// delete out of bound vertices
	for ( int i = 0; i < (int)vertices.size(); i++) {
		// if vertex is not in the image
		//printf("%d %f \n", YDIM, vertices.at(i).y);
		if (vertices.at(i).x <= 0 || vertices.at(i).x >= XDIM || vertices.at(i).y <= 0 || vertices.at(i).y >= YDIM) {
			vertices.erase(vertices.begin(), vertices.begin() + i);
		}
	}
	*/

	// draw edges
	circle(outImg,cvPoint(vertices[0].x,vertices[0].y),4,CV_RGB(0,0,100),-1,8,0);
	for ( int i = 0; i < (int)vertices.size() - 1; i++ ) {
		line(outImg,cvPoint(vertices[i].x,vertices[i].y),cvPoint(vertices[i+1].x,vertices[i+1].y),CV_RGB(100,0,0),2,8,0);
		circle(outImg,cvPoint(vertices[i+1].x,vertices[i+1].y),4,CV_RGB(0,0,100),-1,8,0);
	}
}


void subtructAverage(Mat& current) {

	Scalar average = mean(current);
	
	printf("average ");
	printf("%f ", average.val[0]);
	printf("%f\n", average.val[1]);
	

	float max_x = 0;
	float max_y = 0;
	float min_x = 0;
	float min_y = 0;

	float ini_max_x = 0;
	float ini_max_y = 0;
	float ini_min_x = 0;
	float ini_min_y = 0;

	for ( int row = 0; row < current.rows; row++ ){
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < current.cols; col++){

			if ( ptr->x > ini_max_x )
				ini_max_x = ptr->x;
			if ( ptr->y > ini_max_y )
				ini_max_y = ptr->y;
			if ( ptr->x < ini_min_x )
				ini_min_x = ptr->x;
			if ( ptr->y < ini_min_y )
				ini_min_y = ptr->y;

			float tx = ptr->x - average.val[0];
			float ty = ptr->y - average.val[1];
			

			/*
			if (ptr->x * ptr->x + ptr->y * ptr->y == 0)
			{
				ptr->x = tx;
				ptr->y = ty;
			}
			else 
			{
				float proj = (ptr->x * tx + ptr->y * ty) / (ptr->x * ptr->x + ptr->y * ptr->y);

				ptr->x = ptr->x * proj;
				ptr->y = ptr->y * proj;
			}
			*/


			ptr->x = ptr->x - average.val[0];
			ptr->y = ptr->y - average.val[1];


			if ( ptr->x > max_x )
				max_x = ptr->x;
			if ( ptr->y > max_y )
				max_y = ptr->y;
			if ( ptr->x < min_x )
				min_x = ptr->x;
			if ( ptr->y < min_y )
				min_y = ptr->y;
			
			ptr++;
		}
	}

	
	average = mean(current);
	printf("after  ");
	printf("%f ", average.val[0]);
	printf("%f\n", average.val[1]);
	printf("ini max  ");
	printf("%f ", ini_max_x);
	printf("%f\n", ini_max_y);
	printf("ini min  ");
	printf("%f ", ini_min_x);
	printf("%f\n", ini_min_y);
	printf("max  ");
	printf("%f ", max_x);
	printf("%f\n", max_y);
	printf("min  ");
	printf("%f ", min_x);
	printf("%f\n", min_y);
	
	
}

void subtructMeanMagnitude(Mat& current) {

	Scalar average = mean(current);
	
	printf("average ");
	printf("%f ", average.val[0]);
	printf("%f\n", average.val[1]);
	

	float max_x = 0;
	float max_y = 0;
	float min_x = 0;
	float min_y = 0;

	float ini_max_x = 0;
	float ini_max_y = 0;
	float ini_min_x = 0;
	float ini_min_y = 0;

	float meanval = 0;

	for ( int row = 0; row < current.rows; row++ )
	{
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < current.cols; col++)
		{
			meanval += sqrt(ptr->x * ptr->x + ptr->y * ptr->y);
			ptr++;
		}
	}

	meanval = meanval / (current.rows * current.cols);
	printf("mean %f\n", meanval);

	for ( int row = 0; row < current.rows; row++ ){
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < current.cols; col++){

			if ( ptr->x > ini_max_x )
				ini_max_x = ptr->x;
			if ( ptr->y > ini_max_y )
				ini_max_y = ptr->y;
			if ( ptr->x < ini_min_x )
				ini_min_x = ptr->x;
			if ( ptr->y < ini_min_y )
				ini_min_y = ptr->y;

			float magnitude = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

			float unit_x;
			float unit_y;

			if (magnitude == 0)
			{
				unit_x = 0.0;
				unit_y = 0.0;
			}
			else
			{
				unit_x = ptr->x / magnitude;
				unit_y = ptr->y / magnitude;
			}

			ptr->x = unit_x * (magnitude - meanval);
			ptr->y = unit_y * (magnitude - meanval);



			if ( ptr->x > max_x )
				max_x = ptr->x;
			if ( ptr->y > max_y )
				max_y = ptr->y;
			if ( ptr->x < min_x )
				min_x = ptr->x;
			if ( ptr->y < min_y )
				min_y = ptr->y;
			
			ptr++;
		}
	}

	
	average = mean(current);
	printf("after  ");
	printf("%f ", average.val[0]);
	printf("%f\n", average.val[1]);
	printf("ini max  ");
	printf("%f ", ini_max_x);
	printf("%f\n", ini_max_y);
	printf("ini min  ");
	printf("%f ", ini_min_x);
	printf("%f\n", ini_min_y);
	printf("max  ");
	printf("%f ", max_x);
	printf("%f\n", max_y);
	printf("min  ");
	printf("%f ", min_x);
	printf("%f\n", min_y);

	float aftermeanval = 0;

	for ( int row = 0; row < current.rows; row++ )
	{
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		for ( int col = 0; col < current.cols; col++)
		{
			aftermeanval += sqrt(ptr->x * ptr->x + ptr->y * ptr->y);
			ptr++;
		}
	}

	aftermeanval = aftermeanval / (current.rows * current.cols);
	printf("after mean %f\n", aftermeanval);
	
	
}

void vectorToColor(Mat& current, Mat& outImg) {

	static float max_displacement = 0;
	float max_displacement_new = 0;

	float global_theta = 0;
	float global_magnitude = 0;

	for ( int row = 0; row < current.rows; row++ ) {
		Pixel2* ptr = current.ptr<Pixel2>(row, 0);
		Pixelc* ptr2 = outImg.ptr<Pixelc>(row, 0);

		for ( int col = 0; col < current.cols; col++ ) {
			float theta = atan2(ptr->y, ptr->x)*180/M_PI;	// find angle
			theta += theta < 0 ? 360 : 0;	// enforce strict positive angle
			
			// store vector data
			ptr2->x = theta / 2;
			ptr2->y = 255;
			//ptr2->z = sqrt(ptr->x * ptr->x + ptr->y * ptr->y)*128/max_displacement+128;
            ptr2->z = sqrt(ptr->x * ptr->x + ptr->y * ptr->y)*255/max_displacement;
			// ptr2->z = 255;
			//if ( ptr2->z < 30 ) ptr2->z = 0;

			// store the previous max to maxmin next frame
			if ( sqrt(ptr->x * ptr->x + ptr->y * ptr->y) > max_displacement_new ) max_displacement_new = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);

			global_theta += ptr2->x * ptr2->z;
			global_magnitude += ptr2->z;


			ptr++;
			ptr2++;
		}
	}

	max_displacement = max_displacement_new;

	// show as hsv format
	cvtColor(outImg, outImg, CV_HSV2BGR);
}

void shearRateToColor(Mat& current, Mat& outImg) {

	static float max_displacement = 0.0;
	static float max_frobeniusNorm = 0.0;
	float max_frobeniusNorm_new = 0.0;

	float global_theta = 0;
	float global_magnitude = 0;

	int offset = 10;

	// Iterate through all pixels except for the very edge
	for ( int row = offset; row < current.rows - offset; row++ ) {
		Pixel2* ptr = current.ptr<Pixel2>(row, offset);
		Pixelc* ptr2 = outImg.ptr<Pixelc>(row, offset);

		for ( int col = offset; col < current.cols - offset; col++ ) {

			// obtain the neighbor vectors
			Pixel2 above = current.at<Pixel2>(row-offset, col);
			Pixel2 below = current.at<Pixel2>(row+offset, col);
			Pixel2 left = current.at<Pixel2>(row, col-offset);
			Pixel2 right = current.at<Pixel2>(row, col+offset);

			// Find the velocity gradient matrix
			/*
			/ | dvx/dx dvx/dy |
 			/ | dvy/dx dvy/dy |
			*/
			Mat jacobian = Mat_<Pixel2>(2,2);
			jacobian.at<float>(0,0) = right.x - left.x;
			jacobian.at<float>(0,1) = above.x - below.x;
			jacobian.at<float>(1,0) = right.y - left.y;
			jacobian.at<float>(1,1) = above.y - below.y;

			// printf("%f\n",sqrt(jacobian.dot(jacobian)));

			/*
			Mat jacobianS = Mat_<Pixel2>(2,2);
			jacobianS.at<float>(0.0) = (jacobian.at<float>(0,0) + jacobian.at<float>(0,0)) / 2;
			jacobianS.at<float>(0,1) = (jacobian.at<float>(0,1) + jacobian.at<float>(1,0)) / 2;
			jacobianS.at<float>(1,0) = (jacobian.at<float>(1,0) + jacobian.at<float>(0,1)) / 2;
			jacobianS.at<float>(1,1) = (jacobian.at<float>(1,1) + jacobian.at<float>(1,1)) / 2;
			*/

			//float frobeniusNorm = sqrt(sum(jacobian.mul(jacobian))[0]);
			float frobeniusNorm = jacobian.at<float>(0,0) * jacobian.at<float>(0,0)
							+ jacobian.at<float>(0,1) * jacobian.at<float>(0,1)
							+ jacobian.at<float>(1,0) * jacobian.at<float>(1,0)
							+ jacobian.at<float>(1,1) * jacobian.at<float>(1,1);
			frobeniusNorm = sqrt(frobeniusNorm);

			float theta = atan2(ptr->y, ptr->x)*180/M_PI;	// find angle
			theta += theta < 0 ? 360 : 0;	// enforce strict positive angle
			
			// store vector data
			ptr2->x = 128 - frobeniusNorm*128/max_frobeniusNorm;
			ptr2->y = 255;
            ptr2->z = 255;
			// ptr2->z = 255;
			//if ( ptr2->z < 30 ) ptr2->z = 0;

			// store the previous max to maxmin next frame
			if ( sqrt(ptr->x * ptr->x + ptr->y * ptr->y) > max_displacement ) max_displacement = sqrt(ptr->x * ptr->x + ptr->y * ptr->y);
			max_frobeniusNorm_new = max(frobeniusNorm, max_frobeniusNorm_new);
	
			global_theta += ptr2->x * ptr2->z;
			global_magnitude += ptr2->z;


			ptr++;
			ptr2++;
		}
	}

	max_frobeniusNorm = max_frobeniusNorm_new;

	// show as hsv format
	cvtColor(outImg, outImg, CV_HSV2BGR);
}

PopulationMap::PopulationMap(Pixel2 rectStart, Pixel2 rectEnd, int numberOfVertices) {

	for (int i = 0; i < numberOfVertices; i++)
	{
		sranddev();
		float randX = (rectEnd.x - rectStart.x) * (((double) rand() / (RAND_MAX)) + 1) + rectStart.x;
		float randY = (rectEnd.y - rectStart.y) * (((double) rand() / (RAND_MAX)) + 1) + rectStart.y;
		vertices.push_back(Pixel2(randX, randY));
	}
	printf("%f", vertices[0].x);

}

void PopulationMap::runLK(UMat u_prev, UMat u_current, Mat& outImg) {
	// return status values of calcOpticalFlowPyrLK
	vector<uchar> status;
	vector<float> err;

	// output locations of vertices
	vector<Pixel2> vertices_next;

	// run LK for all vertices
	calcOpticalFlowPyrLK(u_prev, u_current, vertices, vertices_next, status, err, Size(50,50),3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1), 10, 1e-4 );

	/*
	// eliminate any large movement
	for ( int i = 0; i < (int)vertices_next.size(); i++) {
		if ( abs(vertices.at(i).x - vertices_next.at(i).x) > XDIM * 0.1 
			|| abs(vertices.at(i).y - vertices_next.at(i).y) > YDIM * 0.1 ) {
				vertices_next.at(i) = vertices.at(i);
			}
	}
	*/

	// copy the result for the next frame
	vertices = vertices_next;

	/*
	// delete out of bound vertices
	for ( int i = 0; i < (int)vertices.size(); i++) {
		// if vertex is not in the image
		//printf("%d %f \n", YDIM, vertices.at(i).y);
		if (vertices.at(i).x <= 0 || vertices.at(i).x >= XDIM || vertices.at(i).y <= 0 || vertices.at(i).y >= YDIM) {
			vertices.erase(vertices.begin(), vertices.begin() + i);
		}
	}
	*/

	// draw vertices with transparency
	Mat overlay;
	double opacity = 0.5;
	for ( int i = 0; i < (int)vertices.size(); i++ ) {
		outImg.copyTo(overlay);
		circle(overlay,cvPoint(vertices[i].x,vertices[i].y),10,CV_RGB(100,0,0),-1,8,0);
		addWeighted(overlay, opacity, outImg, 1 - opacity, 0, outImg, -1);
	}
}

void drawFrameCount(Mat& outImg, int framecount) {
	putText(outImg, to_string(framecount), cvPoint(30,30), 
	FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
}