#pragma once
#include "Utils/util.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp> 
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

// ============== NOTIFICATION =============== //
//! image no. is encoded from 0, 2, ..., n-1.  //
//!                                            //
// =========================================== //

class PointMatcher
{
public:
	PointMatcher(vector<string> imgNameList, bool extraPoints)
	{
		featureDimension = 64;
		_imgNameList = imgNameList;
		_imgNum = imgNameList.size();
		featureExtractor(extraPoints);
	};

public:
	void featureExtractor(bool extraPoints);
	void readfeatures(int imgIndex, vector<Point2d> &keyPts, Mat &descriptors, double ratio);
	void savefeatures(vector<KeyPoint> keyPts, Mat descriptors, string saveName);
	void loadImgSizeList();
	void saveImgSizeList();

	bool tentativeMatcher(int imgIndex1, int imgIndex2);
	bool featureMatcher(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);    //! all the imgIndex start from 1
	void saveMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> pointSet1, vector<Point2d> pointSet2);
	bool loadMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);

	void pointConvert(Mat_<double> homoMat, Point2d src, Point2d &dst);
	void drawMatches(int imgIndex1, int imgIndex2, vector<Point2d> pointSet1, vector<Point2d> pointSet2);

public:
	vector<Size> _imgSizeList;
	vector<string> _imgNameList;

private:
	int _imgNum;
	int featureDimension;
	vector<string> _keysFileList;
};