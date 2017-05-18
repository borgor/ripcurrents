#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name

#include "pathlines.h"


typedef cv::Point_<float> Pixel2;

int streamline (Pixel2 * pt, cv::Scalar color, cv::Mat flow, cv::Mat overlay, float dt, int iterations){

	Pixel2* flowpt;
	
	for( int i = 0; i< iterations; i++){

		float x = pt->x;
		float y = pt->y;

		int xind = (int) floor(x);
		int yind = (int) floor(y);
		float xrem = x - xind;
		float yrem = y - yind;

		if(xind < 0 || yind < 0 || xind + 1 > flow.cols || yind  + 1 > flow.rows)  //Verify array bounds
		{
			return -1;
		}

		//Bilinear interpolation
		Pixel2 delta =		(*flow.ptr<Pixel2>(yind,xind))		* (1-xrem)*(1-yrem) +
						(*flow.ptr<Pixel2>(yind,xind+1))	* (xrem)*(1-yrem) +
						(*flow.ptr<Pixel2>(yind+1,xind))	* (1-xrem)*(yrem) +
						(*flow.ptr<Pixel2>(yind+1,xind+1))	* (xrem)*(yrem) ;

		Pixel2 newpt = *pt + delta*dt/iterations;
	
		cv::line(overlay,* pt, newpt, color, 1, 8, 0);
	
		*pt = newpt;
	}
		
	return 0;
}
