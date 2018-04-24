#pragma once
#include "Utils/util.h"
#include "featureMatch.h"
#include "graphPro.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp> 
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <fstream>

#define Target_Octave 1         //! select subset of features for building similarity table

using namespace std;
using namespace cv;

struct BiNode
{
	int preNo;
	int curNo;
};

struct Quadra
{
	Point2d centroid;
	Size imgSize;
//	Point2d nodes[4];
	Point2d RME;
};

struct Keys
{
	vector<Point2d> pts;
	Mat descriptors;
};

// ============== NOTIFICATION =============== //
//! node is encoded from 0, 2, ..., n-1.       //
//! cost graph: cost of non-overlap pair is -1 //
// =========================================== //

#define Load_Matches 0        //! just for debugging

class TopoFinder
{
public:
	TopoFinder(PointMatcher *matcher)
	{
		_Ptmatcher = matcher;
		_imgNum = _Ptmatcher->_imgSizeList.size();
		_shotNum = _attempNum = 0;
		_totalTime = _matchTime = 0;
		_isInOrder = false;
	};
	~TopoFinder(){};

public:
	//! return the topology depicted as a cost/weight table
	Mat_<double> findTopology(bool shallLoad, bool isInOrder);
	Mat_<int> loadSimilarityMat();
	void saveSimilarityMat(const Mat_<int> &similarityMat);
	//! similarity value : number of matched features between image pair
	Mat_<int> detectSimilarityOnGlobal();
	Mat_<int> detectSimilarityByGuiding();
	void searchMainChain();
	void buildMainChain();
	//! detect potential overlaps and recalculate aligning model
	void detectPotentialOverlap(int curIndex, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);
	//! vocabulary tree -> association/guiding table
	Mat_<double> getGuidingTable();    //! not implemented
	Mat_<double> getGuidingTableP();   //! select features extracted from the top octave

	int findNodeIndex(int imgNo);
	//! X1 = Model * X2
	Mat_<double> findFastAffine(vector<Point2d> pointSet1, vector<Point2d> pointSet2);
	void loadKeyFiles();
	bool featureMatcher(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2);
	int calSimilarNum(int imgIndex1, int imgIndex2);
	//! temperate functions
	void drawTopoNet();
	Rect TsetImageSize();
	void TsaveMosaicImage();
	void drawTreeLevel();

private:
	int _imgNum;
	Mat_<int> _similarityMat;
	Mat_<uchar> _attempMap;                   //! only for unordered mode : 1 means attempted already
	bool _isInOrder;

	vector<TreeNode> _visitOrder0;           //! the alignment order
	vector<Mat_<double> > _affineMatList;    //! same order with "_visitOrder0"
	vector<Mat_<double> > _covMatrixList; 
	vector<Quadra> _projCoordSet;            //! same order with "_visitOrder0"
	PointMatcher *_Ptmatcher;
	vector<Keys> _keyList;                   //! same with image no.
	vector<vector<int> > _subKeyIndexList;   //! subset for building similarity table   
	//! variables for efficiency analysis
	int _attempNum, _shotNum;
	int _totalTime, _matchTime;
};