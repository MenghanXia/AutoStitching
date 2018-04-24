#pragma once
#include "featureMatch.h"
#include "graphPro.h"
#include "topology.h"
#include "levmar.h"
#include <iostream>
#include <fstream>

#define PENALTY_COEFF 0.10  //! set for LM only
#define Lambada       0.12  //! set for BA only
#define OPT_GROUP_NUM 30
#define BKGRNDPIX     255
#define Need_Mask     1

using namespace std;
using namespace cv;
using namespace Utils;

struct Match_Net
{
	int imgNo;                    //! image no. start from 0
	vector<int> relatedImgs;      //! the position index of overlap-image in visitOrder
	vector<vector<Point2d> > PointSet;
};

struct LMData
{
	vector<Match_Net> *matchPtr;
	vector<Mat_<double> > *modelPtr;
	vector<TreeNode> visitOrder;
	int sIndex;
	int eIndex;
};

//--------------------------------------------------------------------

//! Notification : imgIndex/imgNo start from 0

class ImageAligner
{
public:
	ImageAligner(vector<string> filePathList)
	{
		_filePathList = filePathList;
		_imgNum = filePathList.size();
		_refImgNo = 0;
	};
	~ImageAligner(){};

public:             
	//*** functions for sorting the topological relationship of images ***//
	void sortImageOrder(int referNo, bool shallLoad, bool isInorder);
	void divideImageGroups();
	bool loadMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);

public: 
	//*** functions for building and optimizing the alignment  ***//
	void imageStitcherbyGroup(int referNo);
	void imageStitcherbySolos(int referNo);
	void fillImageMatchNet();
	//! initial alignment by affine model
	void solveGroupModels(int sIndex, int eIndex);
	//! initial alignment by similarity transformation model
	void solveGroupModelsS(int sIndex, int eIndex);
	//! initial alignment by affine model
	void solveSingleModel(int imgIndex);
	int findVisitIndex(int imgNo);
	//! alignment refinement to homographic model : LM Optimization
	void RefineAligningModels(int sIndex, int eIndex);
	void buildIniSolution(double* X, int sIndex, int eIndex);
	static void OptimizationFunction(double* X, double* d, int m, int n, void* data);
	//! alignment refinement to homographic model : Least Square Optimization
	void bundleAdjustingA(int sIndex, int eIndex);  //! method 1 : big matrix
	void bundleAdjustinga(int sIndex, int eIndex);  //! method 1 : normalize by point
	void bundleAdjusting(int sIndex, int eIndex);   //! method 2 : normalize by point
	void buildIniSolution(double* X, double* initX, int sIndex, int eIndex);

	double CalWarpDeviation(vector<Point2d> pointSet1, vector<Point2d> pointSet2, Mat_<double> cvtMat, vector<double> weightList);
	//! detect potential overlapping relationships based on the new position (only used in the refinement)
	void recheckTopology(int sIndex, int eIndex);

	//! display or output functions
	Rect setImageSize(vector<Point2d> &nodePts);
	void saveMosaicImage();         //! aligning in added order
	void saveMosaicImageP();        //! aligning in image no. order
	void outputPrecise();
	void loadHomographies();
	void saveModelParams();
	void drawTopologyNet();
	void labelGroupNodes();
	void drawSimilarMatrix();

private:
	PointMatcher *_matcher;
	Mat_<int> _similarityMat;
	int _refImgNo;
	int _imgNum;
	int _alignedNum;
	vector<int> _groupCusorList;
	vector<TreeNode> _visitOrder;            //! aligning order according this stack
	vector<Quadra> _projCoordSet;            //! same order with "_visitOrder"
	vector<Match_Net> _matchNetList;         //! matching relation of each image (order agree with image no.)
	vector<Mat_<double> > _alignModelList;   //! aligning model parameters of each image (order agree with '_visitOrder')
	vector<Mat_<double> > _initModelList;

	vector<double> reliabilityList;          //! record the mean square error of the initial affine model of each image
	vector<string> _filePathList;
	vector<Size> _imgSizeList;               //! order agree with image no.
};
