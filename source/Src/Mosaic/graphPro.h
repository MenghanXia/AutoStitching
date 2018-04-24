#pragma once
#include "Utils/util.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp> 
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;
using namespace Utils;

// ===================== NOTIFICATION ====================== //
//! node is encoded from 0, 2, ..., n-1.                     //
//! cost graph: all cost must be a positive value, and cost  //
//! of the non-overlap pair is specially set as -1.          //
// ========================================================= //

class Graph
{
public:
	Graph()
	{

	};
	~Graph(){};

	//! single-source shortest path algorithm
	static vector<TreeNode> DijkstraForPath(Mat_<double> graph, int rootNo);
	//! shortest path algorithm for all node pairs
	static vector<TreeNode> FloydForPath(Mat_<double> graph);
	//! minimum spanning tree
	static Mat_<int> extractMSTree(Mat_<double> graph);

	//! tool functions
	static vector<TreeNode> traverseBreadthFirst(Mat_<int> path, int rootNo);
	static void appendix(Mat_<double> dist, int root);
};