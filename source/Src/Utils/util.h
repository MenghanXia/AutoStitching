#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Windows.h>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

namespace Utils
{
	// ********** global variable list ********** //
	static string baseDir = "C:/Users/Richard/Desktop/AutoMosaic/Source/Data/";

	struct TreeNode
	{
		TreeNode(){};
		TreeNode(int im, int re, int le)
		{
			imgNo = im;
			refNo = re;
			level = le;
		};

		int level;       //! the level of node in the tree
		int imgNo;       //! node no.
		int refNo;       //! parent node no.
	};

	vector<string> get_filelist(string foldname);
	Mat_<double> buildCostGraph(const Mat_<int> &similarMat);
	bool loadMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);
	Point2d pointTransform(Mat_<double> homoMat, Point2d srcPt);
	void pointTransform(Mat_<double> homoMat, Point2d srcPt, Point2d &dstPt);
	void pointTransform(Mat_<double> homoMat, vector<Point2d> &pointSet);
	double calPointDist(Point2d point1, Point2d point2);
	double calVecDot(Point2d vec1, Point2d vec2);
	//! convert gray image to pesudo-color image
	Mat grayToPesudoColor(Mat grayMap);
}
