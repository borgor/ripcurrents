#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  //Actually opencv3.2, in spite of the name
#include <cmath>

#include "Streakline.hpp"
#include "ripcurrents.hpp"


typedef cv::Point_<float> Pixel2;

Streakline::Streakline(Pixel2 pixel) {
	// set generation location
	generationPoint = pixel;

	// set the first vertex
	vertices.push_back(pixel);
	numberOfVertices = 1;

	frameCount = 1;
}

void Streakline::runLK(UMat u_prev, UMat u_current, Mat&  outImg) {

	// return status values of calcOpticalFlowPyrLK
	std::vector<uchar> status;
	std::vector<float> err;

	// output locations of vertices
	std::vector<Pixel2> vertices_next;

	// run LK for all vertices
	calcOpticalFlowPyrLK(u_prev, u_current, vertices, vertices_next, status, err, Size(50,50),3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1), 10, 1e-4 );

	// eliminate any large movement
	for ( int i = 0; i < (int)vertices_next.size(); i++) {
		if ( abs(vertices.at(i).x - vertices_next.at(i).x) > XDIM * 0.1 
			|| abs(vertices.at(i).y - vertices_next.at(i).y) > YDIM * 0.1 ) {
				vertices_next.at(i) = vertices.at(i);
			}
	}

	// copy the result for the next frame
	vertices = vertices_next;

	// generate new vertex after certain number of frames
	if (frameCount % 1 == 0) {
		vertices.insert(vertices.begin(), generationPoint);
	}

	// delete out of bound vertices
	for ( int i = 0; i < (int)vertices.size(); i++) {
		// if vertex is not in the image
		//printf("%d %f \n", YDIM, vertices.at(i).y);
		if (vertices.at(i).x <= 0 || vertices.at(i).x >= XDIM || vertices.at(i).y <= 0 || vertices.at(i).y >= YDIM) {
			//vertices.erase(vertices.begin(), vertices.begin() + i);
		}
	}

	// draw generation point
	circle(outImg,Point(generationPoint.x,generationPoint.y),3,CV_RGB(0,100,0),-1,8,0);
	line(outImg,Point(generationPoint.x,generationPoint.y),Point(vertices[0].x,vertices[0].y),CV_RGB(100,0,0),1,8,0);

	// draw edges
	circle(outImg,Point(vertices[0].x,vertices[0].y),2,CV_RGB(0,0,100),-1,8,0);
	for ( int i = 0; i < (int)vertices.size() - 1; i++ ) {
		circle(outImg,Point(vertices[i+1].x,vertices[i+1].y),2,CV_RGB(0,0,100),-1,8,0);
		line(outImg,Point(vertices[i].x,vertices[i].y),Point(vertices[i+1].x,vertices[i+1].y),CV_RGB(100,0,0),1,8,0);
	}

	frameCount++;
}



