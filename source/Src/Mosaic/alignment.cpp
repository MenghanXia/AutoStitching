#include "alignment.h"
#include <omp.h>

//-------------------------------------------------alignment order
void ImageAligner::sortImageOrder(int referNo, bool shallLoad, bool isInorder)
{
	cout<<"#Finding topology ..."<<endl;
	TopoFinder topoBar(_matcher);

	_similarityMat = topoBar.findTopology(shallLoad, isInorder);
	Mat_<double> costGraph = Utils::buildCostGraph(_similarityMat);
	cout<<"-Completed"<<endl;

	cout<<"#Graph analysis for aligning order ..."<<endl;
	if (referNo == -1)
	{
		_visitOrder = Graph::FloydForPath(costGraph);
		_refImgNo = _visitOrder[0].imgNo;
	}
	else
	{
		_visitOrder = Graph::DijkstraForPath(costGraph, referNo);
		_refImgNo = _visitOrder[0].imgNo;
	}
	cout<<"Image "<<_refImgNo<<" become the reference frame."<<endl;
	cout<<"-Completed"<<endl;

	divideImageGroups();
}


void ImageAligner::divideImageGroups()
{
	//! the first group imgIndex
	_groupCusorList.push_back(0);
	//! the following group cursors
	int offset = OPT_GROUP_NUM;
	for (int imgIndex = offset; imgIndex < _imgNum; imgIndex += offset)
	{
		if (_imgNum-imgIndex < offset/2)
		{
			imgIndex = _imgNum-1;
		}
		_groupCusorList.push_back(imgIndex);
	}
	int groupNum = _groupCusorList.size();
	if (_groupCusorList[groupNum-1] < _imgNum-1)
	{
		_groupCusorList.push_back(_imgNum-1);
	}
}


//-------------------------------------------------alignment
void ImageAligner::imageStitcherbyGroup(int referNo)
{
	//! =============== extract features ===============
	bool extractFeature = false; //! ### set this for new data
	_matcher = new PointMatcher(_filePathList, extractFeature);
	_imgSizeList = _matcher->_imgSizeList;

	//! =============== Topology sorting ===============
	bool shallLoad = true, isInOrder = false;     //! ### set this for new data
	sortImageOrder(referNo, shallLoad, isInOrder);

	//! =============== build match net ===============
	fillImageMatchNet();

//	loadHomographies();
	cout<<"#Sequential image alignment start ..."<<endl;
	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);     //cvtMatrix of reference image
	_alignModelList.push_back(identMatrix);
	_initModelList.push_back(identMatrix);
	Quadra bar;
	bar.imgSize = _imgSizeList[_visitOrder[0].imgNo];
	bar.centroid = Point2d(bar.imgSize.width/2, bar.imgSize.height/2);
	_projCoordSet.push_back(bar);
	for (int i = 1; i < _groupCusorList.size(); i ++)
	{
		int sIndex = _groupCusorList[i-1]+1;
		int eIndex = _groupCusorList[i];
		cout<<"Aligning Group "<<i<<endl;
		for (int t = sIndex; t <= eIndex; t ++)
		{
			cout<<_visitOrder[t].imgNo<<" ";
			if ((t-sIndex+1)%10 == 0)
			{
				cout<<endl;
			}
		}
		cout<<"Models initializing ..."<<endl;
		solveGroupModels(sIndex, eIndex);
		cout<<"Done!"<<endl;
//		recheckTopology(sIndex, eIndex);
		cout<<endl;
		bool needRefine = false;
		if (needRefine && i == _groupCusorList.size()-1)
		{
			bundleAdjustingA(1, eIndex);
			//sIndex = 0;
			//RefineAligningModels(sIndex, eIndex);
		}
	}
	cout<<"-Completed!"<<endl;
//	labelGroupNodes();
	saveModelParams();
	drawTopologyNet();
	outputPrecise();
	saveMosaicImageP();
	cout<<"== Mosaic completed successfully!\n";
}


void ImageAligner::imageStitcherbySolos(int referNo)
{
	//! =============== extract features ===============
	bool extractFeature = false;     //! set this for new data
	_matcher = new PointMatcher(_filePathList, extractFeature);
	_imgSizeList = _matcher->_imgSizeList;
	//! =============== Topology sorting ===============
	bool shallLoad = true, isInOrder = false;     //! ### set this for new data
	sortImageOrder(referNo, shallLoad, isInOrder);
	//	return false;
	//! =============== build match net ===============
	fillImageMatchNet();

	cout<<"#Sequential image alignment start ..."<<endl;
	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);     //cvtMatrix of reference image
	_alignModelList.push_back(identMatrix);
	_initModelList.push_back(identMatrix);
	bool needRefine = true;
	for (int i = 1; i < _imgNum; i ++)
	{
		cout<<"Aligning Image "<<_visitOrder[i].imgNo<<"  ";
		solveSingleModel(i);
		if (needRefine && 0)
		{
			for (int j = 0; j < _groupCusorList.size(); j ++)
			{
				if (i == _groupCusorList[j])
				{
					int sIndex = _groupCusorList[j-1]+1;
					int eIndex = _groupCusorList[j];
					bundleAdjustingA(sIndex, eIndex);
					break;
				}
			}
		}
		cout<<"Done!"<<endl;
	}
	if (needRefine)
	{
		cout<<"-Bundle adjustment ..."<<endl;
		bundleAdjustingA(1, _imgNum-1);
	}

	cout<<"-Completed!"<<endl;
	drawTopologyNet();
	outputPrecise();
	saveModelParams();
	saveMosaicImageP();
	cout<<"== Mosaic completed successfully!\n";
}


void ImageAligner::fillImageMatchNet()
{
	cout<<"#Loading topology matching data ..."<<endl;
	//!initialization
	for (int i = 0; i < _imgNum; i ++)
	{
		Match_Net curBar;
		curBar.imgNo = i;
		_matchNetList.push_back(curBar);
	}
	int sum = 0;
	//! fill matching data
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{
			int PtNum = _similarityMat(i,j);
			if (PtNum == 0)
			{
				continue;
			}
			vector<Point2d> PtSet1, PtSet2;
			if (!Utils::loadMatchPts(i,j,PtSet1,PtSet2))
			{
				continue;
			}
			sum += PtSet1.size();
			int indexj = findVisitIndex(j);
			_matchNetList[i].relatedImgs.push_back(indexj);
			_matchNetList[i].PointSet.push_back(PtSet1);
			int indexi = findVisitIndex(i);
			_matchNetList[j].relatedImgs.push_back(indexi);
			_matchNetList[j].PointSet.push_back(PtSet2);
		}
	}
	cout<<"-Completed! - with "<<sum<<" pairs of matches"<<endl;
}


void ImageAligner::solveGroupModels(int sIndex, int eIndex)
{
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				measureNum += pointSet[j].size();
			}
		}
	}
	int paramNum = 6*(eIndex-sIndex+1);
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			int neigIndex = relatedNos[j];
			if (neigIndex > i)
			{
				continue;
			}
			int neigImgNo = _visitOrder[neigIndex].imgNo;
			vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;

			vector<Point2d> curPts, neigPts;
			curPts = pointSet[j];
			//! case 1 : aligning with aligned image
			if (neigIndex < sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						Utils::pointTransform(_alignModelList[neigIndex], neigPts);
						break;
					}
				}
				int fillPos = 6*(i-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos)     = curPts[k].x; A(2*rn,fillPos+1)   = curPts[k].y; A(2*rn,fillPos+2) = 1;
					A(2*rn+1,fillPos+3) = curPts[k].x; A(2*rn+1,fillPos+4) = curPts[k].y; A(2*rn+1,fillPos+5) = 1;
					L(2*rn)   = neigPts[k].x;
					L(2*rn+1) = neigPts[k].y;
					rn ++;
				}
			}
			//! case 2 : aligning with unaligned image
			else if (neigIndex >= sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						break;
					}
				}
				int fillPos1 = 6*(i-sIndex), fillPos2 = 6*(neigIndex-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos1)     = curPts[k].x; A(2*rn,fillPos1+1)   = curPts[k].y; A(2*rn,fillPos1+2) = 1;
					A(2*rn+1,fillPos1+3) = curPts[k].x; A(2*rn+1,fillPos1+4) = curPts[k].y; A(2*rn+1,fillPos1+5) = 1;
					A(2*rn,fillPos2)     = -neigPts[k].x; A(2*rn,fillPos2+1)   = -neigPts[k].y; A(2*rn,fillPos2+2) = -1;
					A(2*rn+1,fillPos2+3) = -neigPts[k].x; A(2*rn+1,fillPos2+4) = -neigPts[k].y; A(2*rn+1,fillPos2+5) = -1;
					L(2*rn)   = 0;
					L(2*rn+1) = 0;
					rn ++;
				}
			}
		}
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	for (int i = 0; i < paramNum; i += 6)
	{
		Mat_<double> affineModel = (Mat_<double>(3,3) << X(i)  , X(i+1), X(i+2),
			                                            X(i+3), X(i+4), X(i+5),
														     0,      0,     1);
		_alignModelList.push_back(affineModel);
		_initModelList.push_back(affineModel);
	}
}


void ImageAligner::solveGroupModelsS(int sIndex, int eIndex)
{
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				measureNum += pointSet[j].size();
			}
		}
	}
	int paramNum = 4*(eIndex-sIndex+1);
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			int neigIndex = relatedNos[j];
			if (neigIndex > i)
			{
				continue;
			}
			int neigImgNo = _visitOrder[neigIndex].imgNo;
			vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;

			vector<Point2d> curPts, neigPts;
			curPts = pointSet[j];
			//! case 1 : aligning with aligned image
			if (neigIndex < sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						Utils::pointTransform(_alignModelList[neigIndex], neigPts);
						break;
					}
				}
				int fillPos = 4*(i-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos)     = curPts[k].x; A(2*rn,fillPos+1)   = -curPts[k].y; A(2*rn,fillPos+2) = 1;
					A(2*rn+1,fillPos) = curPts[k].y; A(2*rn+1,fillPos+1) = curPts[k].x; A(2*rn+1,fillPos+3) = 1;
					L(2*rn)   = neigPts[k].x;
					L(2*rn+1) = neigPts[k].y;
					rn ++;
				}
			}
			//! case 2 : aligning with unaligned image
			else if (neigIndex >= sIndex)
			{
				for (int t = 0; t < neigRelatedNos.size(); t ++)
				{
					if (neigRelatedNos[t] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[t];
						break;
					}
				}
				int fillPos1 = 4*(i-sIndex), fillPos2 = 4*(neigIndex-sIndex);
				for (int k = 0; k < curPts.size(); k ++)
				{
					A(2*rn,fillPos1)     = curPts[k].x; A(2*rn,fillPos1+1)   = -curPts[k].y; A(2*rn,fillPos1+2) = 1;
					A(2*rn+1,fillPos1) = curPts[k].y; A(2*rn+1,fillPos1+1) = curPts[k].x; A(2*rn+1,fillPos1+3) = 1;
					A(2*rn,fillPos2)     = -neigPts[k].x; A(2*rn,fillPos2+1)   = neigPts[k].y; A(2*rn,fillPos2+2) = -1;
					A(2*rn+1,fillPos2) = -neigPts[k].y; A(2*rn+1,fillPos2+1) = -neigPts[k].x; A(2*rn+1,fillPos2+3) = -1;
					L(2*rn)   = 0;
					L(2*rn+1) = 0;
					rn ++;
				}
			}
		}
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	for (int i = 0; i < paramNum; i += 4)
	{
		Mat_<double> affineModel = (Mat_<double>(3,3) << X(i)  , -X(i+1), X(i+2),
			                                             X(i+1), X(i), X(i+3),
			                                             0,      0,     1);
		//		cout<<modelParam<<endl;
		_alignModelList.push_back(affineModel);
		_initModelList.push_back(affineModel);
	}
}


void ImageAligner::solveSingleModel(int imgIndex)
{
	int measureNum = 0;
	int imgNo = _visitOrder[imgIndex].imgNo;
	vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
	vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
	for (int i = 0; i < relatedNos.size(); i ++)
	{
		if (relatedNos[i] < imgIndex)
		{
			measureNum += pointSet[i].size();
		}
	}
	int paramNum = 6;
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	int rn = 0;
	vector<Point2d> PtSet1, PtSet2;
	for (int i = 0; i < relatedNos.size(); i ++)
	{
		int neigIndex = relatedNos[i];
		if (neigIndex > imgIndex)
		{
			continue;
		}
		vector<Point2d> curPts, neigPts;
		curPts = pointSet[i];
		int neigImgNo = _visitOrder[neigIndex].imgNo;
		vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
		for (int t = 0; t < neigRelatedNos.size(); t ++)
		{
			if (neigRelatedNos[t] == imgIndex)
			{
				neigPts = _matchNetList[neigImgNo].PointSet[t];
				Utils::pointTransform(_alignModelList[neigIndex], neigPts);
				break;
			}
		}
		for (int k = 0; k < curPts.size(); k ++)
		{
			A(2*rn,0)   = curPts[k].x; A(2*rn,1)   = curPts[k].y; A(2*rn,2)   = 1;
			A(2*rn+1,3) = curPts[k].x; A(2*rn+1,4) = curPts[k].y; A(2*rn+1,5) = 1;
			L(2*rn)   = neigPts[k].x;
			L(2*rn+1) = neigPts[k].y;
			rn ++;
		}
		//! for homogaphy
		//for (int k = 0; k < curPts.size(); k ++)
		//{
		//	PtSet1.push_back(curPts[k]);
		//	PtSet2.push_back(neigPts[k]);
		//}
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	Mat_<double> affineModel = (Mat_<double>(3,3) << X(0), X(1), X(2),
		                                            X(3), X(4), X(5),
		                                               0,    0,    1);

//	Mat_<double> homoMat = findHomography(PtSet1, PtSet2, CV_RANSAC, 1.5);

	_alignModelList.push_back(affineModel);
	_initModelList.push_back(affineModel);
}


void ImageAligner::bundleAdjusting(int sIndex, int eIndex)
{
	cout<<"Bundle adjusting ...("<<eIndex-sIndex+1<<" images)"<<endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = Lambada;
	int max_iters = 10;

	int rn = 0, ite = 0;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		Mat_<double> AtA = Mat(paramNum, paramNum, CV_64FC1, Scalar(0));
		Mat_<double> AtL = Mat(paramNum, 1, CV_64FC1, Scalar(0));
		for (int i = sIndex; i <= eIndex; i ++)
		{
			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				int num = curPts.size(), n = 0;
				Mat_<double> Ai = Mat(num, paramNum, CV_64FC1, Scalar(0));
				Mat_<double> Li = Mat(num, 1, CV_64FC1, Scalar(0));
				double *AiPtr = (double*)Ai.data;
				double *LiPtr = (double*)Li.data;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];	
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];

						double K1 = 2*(hX0/hW0-x1) + 2*lambada*(hX0/hW0-orgx0);
						double K2 = 2*(hY0/hW0-y1) + 2*lambada*(hY0/hW0-orgy0);

						//! for : x = ...
						AiPtr[n*paramNum+fillPos0]   = K1*x0/hW0; 
						AiPtr[n*paramNum+fillPos0+1] = K1*y0/hW0;
						AiPtr[n*paramNum+fillPos0+2] = K1*1/hW0;
						AiPtr[n*paramNum+fillPos0+3] = K2*x0/hW0;
						AiPtr[n*paramNum+fillPos0+4] = K2*y0/hW0;
						AiPtr[n*paramNum+fillPos0+5] = K2*1/hW0;
						AiPtr[n*paramNum+fillPos0+6] = -(K1+K2)*x0*hX0/(hW0*hW0);
						AiPtr[n*paramNum+fillPos0+7] = -(K1+K2)*y0*hX0/(hW0*hW0);

						double delta_d = (hX0/hW0-x1)*(hX0/hW0-x1) + (hY0/hW0-y1)*(hY0/hW0-y1);
						double delta_r = lambada*((hX0/hW0-orgx0)*(hX0/hW0-orgx0) + (hY0/hW0-orgy0)*(hY0/hW0-orgy0));
						LiPtr[n] = -(delta_d+delta_r);

						double bias = sqrt(delta_d+delta_r);
						meanBias += bias;
						n ++;
						rn ++;
					}
					//! get in normal equation matrix
					Mat_<double> Ait = Ai.t();
					Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
					AtA += barA;
					AtL += barL;
					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1				
					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];					
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];

					double K1 = 2*(hX0/hW0-hX1/hW1) + 2*lambada*(hX0/hW0-orgx0);
					double K2 = 2*(hY0/hW0-hY1/hW1) + 2*lambada*(hY0/hW0-orgy0);
					double K3 = -2*(hX0/hW0-hX1/hW1) + 2*lambada*(hX1/hW1-orgx1);
					double K4 = -2*(hY0/hW0-hY1/hW1) + 2*lambada*(hY1/hW1-orgy1);

					//! for : x = ...
					//! cur-image
					AiPtr[n*paramNum+fillPos0]   = K1*x0/hW0;
					AiPtr[n*paramNum+fillPos0+1] = K1*y0/hW0;
					AiPtr[n*paramNum+fillPos0+2] = K1*1/hW0;
					AiPtr[n*paramNum+fillPos0+3] = K2*x0/hW0;
					AiPtr[n*paramNum+fillPos0+4] = K2*y0/hW0;
					AiPtr[n*paramNum+fillPos0+5] = K2*1/hW0;
					AiPtr[n*paramNum+fillPos0+6] = -(K1+K2)*x0*hX0/(hW0*hW0);
					AiPtr[n*paramNum+fillPos0+7] = -(K1+K2)*y0*hX0/(hW0*hW0);
					//! neig-image
					AiPtr[n*paramNum+fillPos1]   = K3*x1/hW1;
					AiPtr[n*paramNum+fillPos1+1] = K3*y1/hW1;
					AiPtr[n*paramNum+fillPos1+2] = K3*1/hW1;
					AiPtr[n*paramNum+fillPos1+3] = K4*x1/hW1;
					AiPtr[n*paramNum+fillPos1+4] = K4*y1/hW1;
					AiPtr[n*paramNum+fillPos1+5] = K4*1/hW1;
					AiPtr[n*paramNum+fillPos1+6] = -(K3+K4)*x1*hX1/(hW1*hW1);
					AiPtr[n*paramNum+fillPos1+7] = -(K3+K4)*y1*hX1/(hW1*hW1);

					double delta_d = (hX0/hW0-hX1/hW1)*(hX0/hW0-hX1/hW1) + (hY0/hW0-hY1/hW1)*(hY0/hW0-hY1/hW1);
					double delta_r = lambada*((hX0/hW0-orgx0)*(hX0/hW0-orgx0) + (hY0/hW0-orgy0)*(hY0/hW0-orgy0)
						                    + (hX1/hW1-orgx1)*(hX1/hW1-orgx1) + (hY1/hW1-orgy1)*(hY1/hW1-orgy1));
					LiPtr[n] = -(delta_d+delta_r);

					double bias = sqrt(delta_d+delta_r);
					meanBias += bias;
					rn ++;
					n ++;
				}
				//! get in normal equation matrix
				Mat_<double> Ait = Ai.t();
				Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
				AtA += barA;
				AtL += barL;
			}
		}
		meanBias = meanBias/rn;
		cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;
		Mat_<double> dX = AtA.inv()*AtL;
		double *dXPtr = (double*)dX.data;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;
		if (delta < 0.08)
		{
			cout<<"Iteration has converged!"<<endl;
			break;
		}	
		if (ite++ == max_iters)
		{
			cout<<"arrive the limited iterations("<<max_iters<<")"<<endl;
			break;
		}
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}
	delete []X;
	delete []initX;
	cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::bundleAdjustingA(int sIndex, int eIndex)
{
	cout<<"Bundle adjusting ...("<<eIndex-sIndex+1<<" images)"<<endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	Mat_<double> A = Mat(2*measureNum, paramNum, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*measureNum, 1, CV_64FC1, Scalar(0));
	double *APtr = (double*)A.data;
	double *LPtr = (double*)L.data;

	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = Lambada;
	int max_iters = 10;

	int rn = 0, ite = 0;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		for (int i = sIndex; i <= eIndex; i ++)
		{
			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						//! for : x = ...
						//A(2*rn,fillPos0)   = (1+lambada)*x0/hW0;            A(2*rn,fillPos0+1) = (1+lambada)*y0/hW0;            A(2*rn,fillPos0+2) = (1+lambada)*1/hW0;
						//A(2*rn,fillPos0+6) = -(1+lambada)*x0*hX0/(hW0*hW0); A(2*rn,fillPos0+7) = -(1+lambada)*y0*hX0/(hW0*hW0);
						APtr[2*rn*paramNum+fillPos0] = (1+lambada)*x0/hW0; APtr[2*rn*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; APtr[2*rn*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
						APtr[2*rn*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  APtr[2*rn*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
						//L(2*rn) = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);
						LPtr[2*rn] = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);

						//! for : y = ...
						//A(2*rn+1,fillPos0+3) = (1+lambada)*x0/hW0;            A(2*rn+1,fillPos0+4) = (1+lambada)*y0/hW0;            A(2*rn+1,fillPos0+5) = (1+lambada)*1/hW0;
						//A(2*rn+1,fillPos0+6) = -(1+lambada)*x0*hY0/(hW0*hW0); A(2*rn+1,fillPos0+7) = -(1+lambada)*y0*hY0/(hW0*hW0);
						APtr[(2*rn+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
						APtr[(2*rn+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); APtr[(2*rn+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
						//L(2*rn+1) = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);
						LPtr[2*rn+1] = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);

						double bias = (L(2*rn)*L(2*rn) + L(2*rn+1)*L(2*rn+1));
						meanBias += sqrt(bias);
						rn ++;
					}
					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					//! for : x = ...
					//! cur-image
					//A(2*rn,fillPos0)   = (1+lambada)*x0/hW0;            A(2*rn,fillPos0+1) = (1+lambada)*y0/hW0;            A(2*rn,fillPos0+2) = (1+lambada)*1/hW0;
					//A(2*rn,fillPos0+6) = -(1+lambada)*x0*hX0/(hW0*hW0); A(2*rn,fillPos0+7) = -(1+lambada)*y0*hX0/(hW0*hW0);
					APtr[2*rn*paramNum+fillPos0] = (1+lambada)*x0/hW0; APtr[2*rn*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; APtr[2*rn*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
					APtr[2*rn*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  APtr[2*rn*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
					//! neig-image
					//A(2*rn,fillPos1)   = (lambada-1)*x1/hW1;            A(2*rn,fillPos1+1) = (lambada-1)*y1/hW1;            A(2*rn,fillPos1+2) = (lambada-1)*1/hW1;
					//A(2*rn,fillPos1+6) = -(lambada-1)*x1*hX1/(hW1*hW1); A(2*rn,fillPos1+7) = -(lambada-1)*y1*hX1/(hW1*hW1);
					APtr[2*rn*paramNum+fillPos1] = (lambada-1)*x1/hW1; APtr[2*rn*paramNum+fillPos1+1] = (lambada-1)*y1/hW1; APtr[2*rn*paramNum+fillPos1+2] = (lambada-1)*1/hW1;
					APtr[2*rn*paramNum+fillPos1+6] = -(lambada-1)*x1*hX1/(hW1*hW1);  APtr[2*rn*paramNum+fillPos1+7] = -(lambada-1)*y1*hX1/(hW1*hW1);

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];
					//L(2*rn) = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);
					LPtr[2*rn] = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);

					//! for : y = ...
					//! cur-image
					//A(2*rn+1,fillPos0+3) = (1+lambada)*x0/hW0;            A(2*rn+1,fillPos0+4) = (1+lambada)*y0/hW0;            A(2*rn+1,fillPos0+5) = (1+lambada)*1/hW0;
					//A(2*rn+1,fillPos0+6) = -(1+lambada)*x0*hY0/(hW0*hW0); A(2*rn+1,fillPos0+7) = -(1+lambada)*y0*hY0/(hW0*hW0);
					APtr[(2*rn+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; APtr[(2*rn+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
					APtr[(2*rn+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); APtr[(2*rn+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
					//! neig-image
					//A(2*rn+1,fillPos1+3) = (lambada-1)*x1/hW1;            A(2*rn+1,fillPos1+4) = (lambada-1)*y1/hW1;            A(2*rn+1,fillPos1+5) = (lambada-1)*1/hW1;
					//A(2*rn+1,fillPos1+6) = -(lambada-1)*x1*hY1/(hW1*hW1); A(2*rn+1,fillPos1+7) = -(lambada-1)*y1*hY1/(hW1*hW1);
					APtr[(2*rn+1)*paramNum+fillPos1+3] = (lambada-1)*x1/hW1; APtr[(2*rn+1)*paramNum+fillPos1+4] = (lambada-1)*y1/hW1; APtr[(2*rn+1)*paramNum+fillPos1+5] = (lambada-1)*1/hW1;
					APtr[(2*rn+1)*paramNum+fillPos1+6] = -(lambada-1)*x1*hY1/(hW1*hW1); APtr[(2*rn+1)*paramNum+fillPos1+7] = -(lambada-1)*y1*hY1/(hW1*hW1);

					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];
					//L(2*rn+1) = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);
					LPtr[2*rn+1] = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);

					double bias = (L(2*rn)*L(2*rn) + L(2*rn+1)*L(2*rn+1));
					meanBias += sqrt(bias);
					rn ++;
				}
			}
		}
		meanBias = meanBias/rn;
		cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;
		Mat_<double> At = A.t();
		Mat_<double> dX = (At*A).inv()*(At*L);
		//		cout<<A.t()*A<<endl<<(A.t()*L)<<endl;
		double *dXPtr = (double*)dX.data;
		//		cout<<dX<<endl;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
				//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;
		if (delta < 0.08)
		{
			cout<<"Iteration has converged!"<<endl;
			break;
		}	
		if (ite++ == max_iters)
		{
			cout<<"arrive the limited iterations("<<max_iters<<")"<<endl;
			break;
		}
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}
	delete []X;
	delete []initX;
	cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::bundleAdjustinga(int sIndex, int eIndex)
{
	cout<<"Bundle adjusting ...("<<eIndex-sIndex+1<<" images)"<<endl;
	int measureNum = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
		vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)     //! avoid repeating counting
			{
				int num = pointSet[j].size();
				num = num%3 == 0 ? (num/3) : (num/3+1);
				measureNum += num;     //! only 1/3 of matching pairs for optimization
			}
		}
	}
	int paramNum = 8*(eIndex-sIndex+1);    //! optimizing homgraphic model with 8 DoF
	double *X = new double[paramNum];
	double *initX = new double[6*(eIndex-sIndex+1)];
	buildIniSolution(X, initX, sIndex, eIndex);
	//! parameters setting of least square optimization
	double lambada = Lambada;
	int max_iters = 10;

	int rn = 0, ite = 0;
	while (1)
	{
		double meanBias = 0;
		rn = 0;
		Mat_<double> AtA = Mat(paramNum, paramNum, CV_64FC1, Scalar(0));
		Mat_<double> AtL = Mat(paramNum, 1, CV_64FC1, Scalar(0));
		for (int i = sIndex; i <= eIndex; i ++)
		{
			//! prepare relative data or parameters of current image
			int imgNo = _visitOrder[i].imgNo;
			vector<vector<Point2d> > pointSet = _matchNetList[imgNo].PointSet;
			vector<int> relatedNos = _matchNetList[imgNo].relatedImgs;
			for (int j = 0; j < relatedNos.size(); j ++)
			{
				int neigIndex = relatedNos[j];
				if (neigIndex > i)
				{
					continue;
				}
				vector<Point2d> curPts, neigPts;
				curPts = pointSet[j];
				int neigImgNo = _visitOrder[neigIndex].imgNo;
				vector<int> neigRelatedNos = _matchNetList[neigImgNo].relatedImgs;
				for (int k = 0; k < neigRelatedNos.size(); k ++)
				{
					if (neigRelatedNos[k] == i)
					{
						neigPts = _matchNetList[neigImgNo].PointSet[k];
						break;
					}
				}

				int curse0 = i-sIndex, curse1 = neigIndex-sIndex;
				int fillPos0 = curse0*8, fillPos1 = curse1*8;
				int num = curPts.size(), n = 0;
				Mat_<double> Ai = Mat(2*num, paramNum, CV_64FC1, Scalar(0));
				Mat_<double> Li = Mat(2*num, 1, CV_64FC1, Scalar(0));
				double *AiPtr = (double*)Ai.data;
				double *LiPtr = (double*)Li.data;
				//! case 1 : with a fixed image
				if (neigIndex < sIndex)
				{
					Utils::pointTransform(_alignModelList[neigIndex], neigPts);
					for (int t = 0; t < curPts.size(); t += 3)
					{
						int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;		
						double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
						double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
						double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

						//! for : x = ...
						AiPtr[2*n*paramNum+fillPos0]   = (1+lambada)*x0/hW0; 
						AiPtr[2*n*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; 
						AiPtr[2*n*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
						AiPtr[2*n*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  
						AiPtr[2*n*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
						double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];

						LiPtr[2*n] = lambada*(orgx0)+x1 - ((1+lambada)*hX0/hW0);

						//! for : y = ...
						AiPtr[(2*n+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0;
						AiPtr[(2*n+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; 
						AiPtr[(2*n+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
						AiPtr[(2*n+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); 
						AiPtr[(2*n+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
						double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];

						LiPtr[2*n+1] = lambada*(orgy0)+y1 - ((1+lambada)*hY0/hW0);

						double bias = (LiPtr[2*n]*LiPtr[2*n] + LiPtr[2*n+1]*LiPtr[2*n+1]);
						meanBias += sqrt(bias);
						n ++;
						rn ++;
					}
					//! get in normal equation matrix
					Mat_<double> Ait = Ai.t();
					Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
					AtA += barA;
					AtL += barL;

					continue;
				}

				//! case 2 : with a remain optimized image
				for (int t = 0; t < curPts.size(); t += 3)
				{
					int x0 = curPts[t].x, y0 = curPts[t].y, x1 = neigPts[t].x, y1 = neigPts[t].y;			
					double hX0 = X[fillPos0+0]*x0 + X[fillPos0+1]*y0 + X[fillPos0+2];     //! h1*x0 + h2*y0 + h3
					double hY0 = X[fillPos0+3]*x0 + X[fillPos0+4]*y0 + X[fillPos0+5];     //! h4*x0 + h5*y0 + h6
					double hW0 = X[fillPos0+6]*x0 + X[fillPos0+7]*y0 + 1;                 //! h7*x0 + h8*y0 + 1

					double hX1 = X[fillPos1+0]*x1 + X[fillPos1+1]*y1 + X[fillPos1+2];     //! h1'*x1 + h2'*y1 + h3'
					double hY1 = X[fillPos1+3]*x1 + X[fillPos1+4]*y1 + X[fillPos1+5];     //! h4'*x1 + h5'*y1 + h6'
					double hW1 = X[fillPos1+6]*x1 + X[fillPos1+7]*y1 + 1;                 //! h7'*x1 + h8'*y1 + 1

					//! for : x = ...
					//! cur-image
					AiPtr[2*n*paramNum+fillPos0]   = (1+lambada)*x0/hW0; 
					AiPtr[2*n*paramNum+fillPos0+1] = (1+lambada)*y0/hW0; 
					AiPtr[2*n*paramNum+fillPos0+2] = (1+lambada)*1/hW0;
					AiPtr[2*n*paramNum+fillPos0+6] = -(1+lambada)*x0*hX0/(hW0*hW0);  
					AiPtr[2*n*paramNum+fillPos0+7] = -(1+lambada)*y0*hX0/(hW0*hW0);
					//! neig-image
					AiPtr[2*n*paramNum+fillPos1]   = (lambada-1)*x1/hW1; 
					AiPtr[2*n*paramNum+fillPos1+1] = (lambada-1)*y1/hW1; 
					AiPtr[2*n*paramNum+fillPos1+2] = (lambada-1)*1/hW1;
					AiPtr[2*n*paramNum+fillPos1+6] = -(lambada-1)*x1*hX1/(hW1*hW1);  
					AiPtr[2*n*paramNum+fillPos1+7] = -(lambada-1)*y1*hX1/(hW1*hW1);

					double orgx0 = initX[6*curse0+0]*x0 + initX[6*curse0+1]*y0 + initX[6*curse0+2];
					double orgx1 = initX[6*curse1+0]*x1 + initX[6*curse1+1]*y1 + initX[6*curse1+2];

					LiPtr[2*n] = lambada*(orgx0+orgx1) - ((1+lambada)*hX0/hW0 + (lambada-1)*hX1/hW1);

					//! for : y = ...
					//! cur-image
					AiPtr[(2*n+1)*paramNum+fillPos0+3] = (1+lambada)*x0/hW0; 
					AiPtr[(2*n+1)*paramNum+fillPos0+4] = (1+lambada)*y0/hW0; 
					AiPtr[(2*n+1)*paramNum+fillPos0+5] = (1+lambada)*1/hW0;
					AiPtr[(2*n+1)*paramNum+fillPos0+6] = -(1+lambada)*x0*hY0/(hW0*hW0); 
					AiPtr[(2*n+1)*paramNum+fillPos0+7] = -(1+lambada)*y0*hY0/(hW0*hW0);
					//! neig-image
					AiPtr[(2*n+1)*paramNum+fillPos1+3] = (lambada-1)*x1/hW1; 
					AiPtr[(2*n+1)*paramNum+fillPos1+4] = (lambada-1)*y1/hW1; 
					AiPtr[(2*n+1)*paramNum+fillPos1+5] = (lambada-1)*1/hW1;
					AiPtr[(2*n+1)*paramNum+fillPos1+6] = -(lambada-1)*x1*hY1/(hW1*hW1); 
					AiPtr[(2*n+1)*paramNum+fillPos1+7] = -(lambada-1)*y1*hY1/(hW1*hW1);

					double orgy0 = initX[6*curse0+3]*x0 + initX[6*curse0+4]*y0 + initX[6*curse0+5];
					double orgy1 = initX[6*curse1+3]*x1 + initX[6*curse1+4]*y1 + initX[6*curse1+5];

					LiPtr[2*n+1] = lambada*(orgy0+orgy1) - ((1+lambada)*hY0/hW0 + (lambada-1)*hY1/hW1);

					double bias = (LiPtr[2*n]*LiPtr[2*n] + LiPtr[2*n+1]*LiPtr[2*n+1]);
					meanBias += sqrt(bias);
					n ++;
					rn ++;
				}
				//! get in normal equation matrix
				Mat_<double> Ait = Ai.t();
				Mat_<double> barA = Ait*Ai, barL = Ait*Li;	
				AtA += barA;
				AtL += barL;
			}
		}
		meanBias = meanBias/rn;
		cout<<"Iteration: "<<ite<<" with cost: "<<meanBias<<endl;
		Mat_<double> dX = AtA.inv()*AtL;
		double *dXPtr = (double*)dX.data;
		//		cout<<dX<<endl;
		double delta = 0;      //! record the translation parameters of images
		int num = 0;
		for (int i = 0; i < paramNum; i ++)
		{
			X[i] += dXPtr[i];
			if ((i+1)%8 == 3 || (i+1)%8 == 6)
			{
				//				cout<<dX(i)<<endl;
				delta += abs(dXPtr[i]);
				num ++;
			}
		}
		delta = delta/num;
		if (delta < 0.08)
		{
			cout<<"Iteration has converged!"<<endl;
			break;
		}	
		if (ite++ == max_iters)
		{
			cout<<"arrive the limited iterations("<<max_iters<<")"<<endl;
			break;
		}
	}
	//! update the optimized parameters
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		double *data = (double*)_alignModelList[i].data;
		for (int j = 0; j < 8; j ++)
		{
			data[j] = X[cnt++];
		}
	}
	delete []X;
	delete []initX;
	cout<<"This optimization round is over!"<<endl;
}


void ImageAligner::buildIniSolution(double* X, double* initX, int sIndex, int eIndex)
{
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _alignModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 8; j ++)
		{
			X[cnt++] = data[j];
		}
	}
	cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _initModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 6; j ++)
		{
			initX[cnt++] = data[j];
		}
	}
}


void ImageAligner::buildIniSolution(double* X, int sIndex, int eIndex)
{
	int cnt = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		Mat_<double> tempMat = _alignModelList[i];
		double *data = (double*)tempMat.data;
		for (int j = 0; j < 8; j ++)
		{
			X[cnt++] = data[j];
		}
	}
}


void ImageAligner::RefineAligningModels(int sIndex, int eIndex)
{
	int m = 0, n = 0, max_its = 200;
	m = (eIndex-sIndex+1) * 8;       //without optimizing the start one
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int curNo = _visitOrder[i].imgNo;
		vector<vector<Point2d> > pointSet = _matchNetList[curNo].PointSet;
		vector<int> relatedNos = _matchNetList[curNo].relatedImgs;
		for (int j = 0; j < relatedNos.size(); j ++)
		{
			if (relatedNos[j] < i)
			{
				//! using only one third of corresponding for optimization
				n += pointSet[j].size()/3;
			}
		}
	}
	double *d = new double[n];
	for (int i = 0; i < n; i ++)
	{
		d[i] = 0;
	}
	double *X = new double[m];
	buildIniSolution(X, sIndex, eIndex);

	LMData *LMInput = new LMData;        //! prepare input data for optimization
	LMInput->sIndex = sIndex;
	LMInput->eIndex = eIndex;
	LMInput->matchPtr = &_matchNetList;
	LMInput->modelPtr = &_alignModelList;
	LMInput->visitOrder = _visitOrder;

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=1E-20; opts[1]=1E-30; opts[2]=1E-30; opts[3]=1E-30;	opts[4]= 1e-9;
	int ret = dlevmar_dif(OptimizationFunction, X, d, m, n, max_its, opts, info, NULL, NULL, (void*)LMInput);
	for (int i = sIndex; i <= eIndex; i ++)    //stock optimized homographic matrix
	{
		double *homoVec = X + 8*(i-sIndex);
		Mat_<double> homoMat(3,3,CV_64FC1);
		homoMat(0,0) = homoVec[0]; homoMat(0,1) = homoVec[1]; homoMat(0,2) = homoVec[2];
		homoMat(1,0) = homoVec[3]; homoMat(1,1) = homoVec[4]; homoMat(1,2) = homoVec[5];
		homoMat(2,0) = homoVec[6]; homoMat(2,1) = homoVec[7]; homoMat(2,2) = 1.0;
		_alignModelList[i] = homoMat;
	}

	delete []d;      //free stack
	delete []X;
	delete LMInput;
	cout<<"Optimization done with iterations("<<ret<<")"<<endl;
}


void ImageAligner::OptimizationFunction(double* X, double* d, int m, int n, void* data)
{
	LMData *dataPtr = (LMData *)data;
	vector<Match_Net> *matchNetPtr = dataPtr->matchPtr;
	int startIndex = dataPtr->sIndex, endIndex = dataPtr->eIndex;
	vector<Mat_<double> > *modelListPtr = dataPtr->modelPtr;
	vector<TreeNode> visitOrder = dataPtr->visitOrder;
	int cnt = 0;
	double meanError = 0;
	for (int i = startIndex; i <= endIndex; i ++)       //without the start image
	{
		int curNo = visitOrder[i].imgNo;
		Match_Net matchNet = (*matchNetPtr)[curNo];
		vector<int> relatedImgs = matchNet.relatedImgs;
		double *homoVec1, *homoVec2;
		double *iniHomoVec1, *iniHomoVec2;
		homoVec1 = X + 8*(i-startIndex);
		iniHomoVec1 = (double*)(*modelListPtr)[i].data;
		for (int j = 0; j < relatedImgs.size(); j ++)
		{
			int neigIndex = relatedImgs[j];
			if (neigIndex > i)
			{
				continue;
			}
			vector<Point2d> ptSet1, ptSet2;
			ptSet1 = matchNet.PointSet[j];         //! points on cur_image
			int neigNo = visitOrder[neigIndex].imgNo;
			Match_Net neigMatchNet = (*matchNetPtr)[neigNo];
			for (int k = 0; k < neigMatchNet.relatedImgs.size(); k ++)
			{
				if (neigMatchNet.relatedImgs[k] == i)
				{
					ptSet2 = neigMatchNet.PointSet[k];
					break;
				}
			}
			bool fixedNeigh = false;
			if (neigIndex < startIndex)
			{
				double *data = (double*)(*modelListPtr)[neigIndex].data;
				homoVec2 = data;
				fixedNeigh = true;
			}
			else
			{
				homoVec2 = X + 8*(neigIndex-startIndex);
				iniHomoVec2 = (double*)(*modelListPtr)[neigIndex].data;
			}
			//! using only one third of corresponding for optimization
			for (int t = 0; t < ptSet1.size(); t += 3)
			{
				double x1 = ptSet1[t].x, y1 = ptSet1[t].y;
				double x2 = ptSet2[t].x, y2 = ptSet2[t].y;
				double mosaic_x1 = (homoVec1[0]*x1 + homoVec1[1]*y1 + homoVec1[2])/(homoVec1[6]*x1 + homoVec1[7]*y1 + 1.0);
				double mosaic_y1 = (homoVec1[3]*x1 + homoVec1[4]*y1 + homoVec1[5])/(homoVec1[6]*x1 + homoVec1[7]*y1 + 1.0);
				double mosaic_x2 = (homoVec2[0]*x2 + homoVec2[1]*y2 + homoVec2[2])/(homoVec2[6]*x2 + homoVec2[7]*y2 + 1.0);
				double mosaic_y2 = (homoVec2[3]*x2 + homoVec2[4]*y2 + homoVec2[5])/(homoVec2[6]*x2 + homoVec2[7]*y2 + 1.0);
				double bias = sqrt((mosaic_x1-mosaic_x2)*(mosaic_x1-mosaic_x2)+(mosaic_y1-mosaic_y2)*(mosaic_y1-mosaic_y2));

				//! perspective penalty items
				double penalty = 0;
				if (!fixedNeigh)
				{
					double iniMosaic_x1 = (iniHomoVec1[0]*x1 + iniHomoVec1[1]*y1 + iniHomoVec1[2])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_y1 = (iniHomoVec1[3]*x1 + iniHomoVec1[4]*y1 + iniHomoVec1[5])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_x2 = (iniHomoVec2[0]*x2 + iniHomoVec2[1]*y2 + iniHomoVec2[2])/(iniHomoVec2[6]*x2 + iniHomoVec2[7]*y2 + 1.0);
					double iniMosaic_y2 = (iniHomoVec2[3]*x2 + iniHomoVec2[4]*y2 + iniHomoVec2[5])/(iniHomoVec2[6]*x2 + iniHomoVec2[7]*y2 + 1.0);
					double penalty1 = sqrt((mosaic_x1-iniMosaic_x1)*(mosaic_x1-iniMosaic_x1)+(mosaic_y1-iniMosaic_y1)*(mosaic_y1-iniMosaic_y1));
					double penalty2 = sqrt((mosaic_x2-iniMosaic_x2)*(mosaic_x2-iniMosaic_x2)+(mosaic_y2-iniMosaic_y2)*(mosaic_y2-iniMosaic_y2));
					penalty = (penalty1 + penalty2)/2;
				}
				else
				{
					double iniMosaic_x1 = (iniHomoVec1[0]*x1 + iniHomoVec1[1]*y1 + iniHomoVec1[2])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					double iniMosaic_y1 = (iniHomoVec1[3]*x1 + iniHomoVec1[4]*y1 + iniHomoVec1[5])/(iniHomoVec1[6]*x1 + iniHomoVec1[7]*y1 + 1.0);
					penalty = sqrt((mosaic_x1-iniMosaic_x1)*(mosaic_x1-iniMosaic_x1)+(mosaic_y1-iniMosaic_y1)*(mosaic_y1-iniMosaic_y1));
				}
				d[cnt++] = bias + PENALTY_COEFF*penalty;
				meanError += bias;
			}
		}
	}
	meanError /= cnt;
	cout<<"current mean-warping-bias is: "<<meanError<<endl;
}


int ImageAligner::findVisitIndex(int imgNo)
{
	int imgIndex = 0;
	for (int i = 0; i < _imgNum; i ++)
	{
		if (_visitOrder[i].imgNo == imgNo)
		{
			imgIndex = i;
		}
	}
	return imgIndex;
}


Rect ImageAligner::setImageSize(vector<Point2d> &nodePts)
{
	vector<Point2d> marginPtList;
	int i, j;
	for (i = 0; i < _imgNum; i ++)
	{
		Mat_<double> homoMat = _alignModelList[i];
		int curImgNo = _visitOrder[i].imgNo;
		Size imgSize = _imgSizeList[curImgNo];
		int height = imgSize.height, width = imgSize.width;
		Point2d srcPt00(0,0), srcPt01(width,0), srcPt10(0,height), srcPt11(width,height);
		Point2d dstPt00, dstPt01, dstPt10, dstPt11;
		Utils::pointTransform(homoMat, srcPt00, dstPt00);
		Utils::pointTransform(homoMat, srcPt01, dstPt01);
		Utils::pointTransform(homoMat, srcPt10, dstPt10);
		Utils::pointTransform(homoMat, srcPt11, dstPt11);
		marginPtList.push_back(dstPt00);
		marginPtList.push_back(dstPt01);
		marginPtList.push_back(dstPt10);
		marginPtList.push_back(dstPt11);
		if (i != 0)
		{
			nodePts.push_back(dstPt10);
			nodePts.push_back(dstPt11);
		}
	}
	int minX = 999, minY = 999, maxX = 0, maxY = 0;
	for (i = 0; i < marginPtList.size(); i ++)
	{
		Point2d tmpPt = marginPtList[i];
		int x = int(fabs(tmpPt.x)+1)*(tmpPt.x < 0 ? -1 : 1);
		int y = int(fabs(tmpPt.y)+1)*(tmpPt.y < 0 ? -1 : 1);
		if (x > maxX)
			maxX = x;
		if (x < minX)
			minX = x;
		if (y > maxY)
			maxY = y;
		if (y < minY)
			minY = y;
	}
	for (i = 0; i < nodePts.size(); i ++)
	{
		nodePts[i].x -= minX;
		nodePts[i].y -= minY;
	}
	Rect mosaicRect;
	mosaicRect.x = minX; mosaicRect.y = minY;
	mosaicRect.width = maxX-minX+1; mosaicRect.height = maxY-minY+1;
	return mosaicRect;
}


void ImageAligner::saveMosaicImage()
{
	cout<<"#Warping sequential images ..."<<endl;
	bool needMask = false, needAlpha = false;
	vector<Point2d> nodePts;
	Rect mosaicRect = setImageSize(nodePts);
	int newRow = mosaicRect.height, newCol = mosaicRect.width;
	int i, j;
	Rect newImgRect;
	Mat stitchImage(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
	uchar *mosaicData = (uchar*)stitchImage.data;
	for (i = 0; i < _imgNum; i ++)
	{
		int curImgNo = _visitOrder[i].imgNo;
		cout<<"Warping Image: "<<curImgNo<<"..."<<endl;
		Mat_<double> homoMat = _alignModelList[i];
		Size imgSize = _imgSizeList[curImgNo];
		int height = imgSize.height, width = imgSize.width;
		Point2d srcPt00(0,0), srcPt01(width,0), srcPt10(0,height), srcPt11(width,height);
		Point2d dstPt00, dstPt01, dstPt10, dstPt11;
		Utils::pointTransform(homoMat, srcPt00, dstPt00);
		Utils::pointTransform(homoMat, srcPt01, dstPt01);
		Utils::pointTransform(homoMat, srcPt10, dstPt10);
		Utils::pointTransform(homoMat, srcPt11, dstPt11);

		double fminX, fminY, fmaxX, fmaxY;
		fminX = min(min(dstPt00.x, dstPt01.x), min(dstPt10.x, dstPt11.x));
		fmaxX = max(max(dstPt00.x, dstPt01.x), max(dstPt10.x, dstPt11.x));
		fminY = min(min(dstPt00.y, dstPt01.y), min(dstPt10.y, dstPt11.y));
		fmaxY = max(max(dstPt00.y, dstPt01.y), max(dstPt10.y, dstPt11.y));
		int minX, minY, maxX, maxY;
		minX = int(fabs(fminX)+1)*(fminX < 0 ? -1 : 1);
		maxX = int(fabs(fmaxX)+1)*(fmaxX < 0 ? -1 : 1);
		minY = int(fabs(fminY)+1)*(fminY < 0 ? -1 : 1);
		maxY = int(fabs(fmaxY)+1)*(fmaxY < 0 ? -1 : 1);

		int startX = minX-mosaicRect.x; int endX = startX+maxX-minX;
		int startY = minY-mosaicRect.y; int endY = startY+maxY-minY;
		int r, c;
		Mat warpedImage;
		if (needMask && needAlpha)
		{
			warpedImage = Mat(newRow, newCol, CV_8UC4, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX,0));
		}
		else if (needMask && !needAlpha)
		{
			warpedImage = Mat(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
		}
		uchar *curWarpData = (uchar*)warpedImage.data;

		string filePath = _filePathList[curImgNo];
		Mat image = imread(filePath);
		uchar *curImgData = (uchar*)image.data;
		Mat_<double> invHomoMat = homoMat.inv();
		for (r = startY; r < endY; r ++)            
		{
			for (c = startX; c < endX; c ++)
			{
				int grayValueR, grayValueG, grayValueB;
				Point2d dstPt(c+mosaicRect.x,r+mosaicRect.y), srcPt(0,0);
				Utils::pointTransform(invHomoMat, dstPt, srcPt);
				int u = int(srcPt.x), v = int(srcPt.y);
				if (0 < u && width-1 > u && 0 < v && height-1 > v)
				{
					int grayValueR1 = 0, grayValueR2 = 0;
					int grayValueG1 = 0, grayValueG2 = 0;
					int grayValueB1 = 0, grayValueB2 = 0;

					//bilinear interpolation
					grayValueR1 = curImgData[3*(v*width+u)+0]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+0]*(srcPt.x-u);
					grayValueG1 = curImgData[3*(v*width+u)+1]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+1]*(srcPt.x-u);
					grayValueB1 = curImgData[3*(v*width+u)+2]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+2]*(srcPt.x-u);

					grayValueR2 = curImgData[3*((v+1)*width+u)+0]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+0]*(srcPt.x-u);
					grayValueG2 = curImgData[3*((v+1)*width+u)+1]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+1]*(srcPt.x-u);
					grayValueB2 = curImgData[3*((v+1)*width+u)+2]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+2]*(srcPt.x-u);

					grayValueR = grayValueR1*(1-(srcPt.y-v)) + grayValueR2*(srcPt.y-v);
					grayValueG = grayValueG1*(1-(srcPt.y-v)) + grayValueG2*(srcPt.y-v);
					grayValueB = grayValueB1*(1-(srcPt.y-v)) + grayValueB2*(srcPt.y-v);

					if (needMask)
					{
						if (!needAlpha)
						{
							curWarpData[3*(r*newCol+c)+0] = grayValueR;
							curWarpData[3*(r*newCol+c)+1] = grayValueG;
							curWarpData[3*(r*newCol+c)+2] = grayValueB;
						}
						else
						{
							warpedImage.at<Vec4b>(r,c)[3] = 255;   //! for alpha channel
							warpedImage.at<Vec4b>(r,c)[0] = grayValueR;  //! B
							warpedImage.at<Vec4b>(r,c)[1] = grayValueG;  //! G
							warpedImage.at<Vec4b>(r,c)[2] = grayValueB;  //! R
						}
					}
					//! set for the mosaic image
					mosaicData[3*(r*newCol+c)+0] = grayValueR;
					mosaicData[3*(r*newCol+c)+1] = grayValueG;
					mosaicData[3*(r*newCol+c)+2] = grayValueB;
				}
			}
		}
		if (!needMask)
		{
			continue;
		}
		char name[512];
		sprintf(name,"/Masks/warp%03d.png", curImgNo);
		string savePath = Utils::baseDir + string(name);
		imwrite(savePath, warpedImage);
	}

	string filePath = Utils::baseDir + "/mosaic.png";
	imwrite(filePath, stitchImage);
	cout<<"-Completed!"<<endl;
}


void ImageAligner::saveMosaicImageP()
{
	cout<<"#Warping sequential images ..."<<endl;
	bool needMask = Need_Mask, needAlpha = false;
	vector<Point2d> nodePts;
	Rect mosaicRect = setImageSize(nodePts);
	int newRow = mosaicRect.height, newCol = mosaicRect.width;
	int i, j;
	Rect newImgRect;
	Mat stitchImage(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
	uchar *mosaicData = (uchar*)stitchImage.data;
	Rect refRect;
	for (i = 0; i < _imgNum; i ++)
	{
		int curImgNo = i;
		int curIndex = findVisitIndex(curImgNo);
		cout<<"Warping Image: "<<curImgNo<<"..."<<endl;
		Mat_<double> homoMat = _alignModelList[curIndex];
		Size imgSize = _imgSizeList[curImgNo];
		int height = imgSize.height, width = imgSize.width;
		Point2d srcPt00(0,0), srcPt01(width,0), srcPt10(0,height), srcPt11(width,height);
		Point2d dstPt00, dstPt01, dstPt10, dstPt11;
		Utils::pointTransform(homoMat, srcPt00, dstPt00);
		Utils::pointTransform(homoMat, srcPt01, dstPt01);
		Utils::pointTransform(homoMat, srcPt10, dstPt10);
		Utils::pointTransform(homoMat, srcPt11, dstPt11);

		double fminX, fminY, fmaxX, fmaxY;
		fminX = min(min(dstPt00.x, dstPt01.x), min(dstPt10.x, dstPt11.x));
		fmaxX = max(max(dstPt00.x, dstPt01.x), max(dstPt10.x, dstPt11.x));
		fminY = min(min(dstPt00.y, dstPt01.y), min(dstPt10.y, dstPt11.y));
		fmaxY = max(max(dstPt00.y, dstPt01.y), max(dstPt10.y, dstPt11.y));
		int minX, minY, maxX, maxY;
		minX = int(fabs(fminX)+1)*(fminX < 0 ? -1 : 1);
		maxX = int(fabs(fmaxX)+1)*(fmaxX < 0 ? -1 : 1);
		minY = int(fabs(fminY)+1)*(fminY < 0 ? -1 : 1);
		maxY = int(fabs(fmaxY)+1)*(fmaxY < 0 ? -1 : 1);

		int startX = minX-mosaicRect.x; int endX = startX+maxX-minX;
		int startY = minY-mosaicRect.y; int endY = startY+maxY-minY;
		if (i == _refImgNo)
		{
			refRect.x = startX; refRect.y = startY;
			refRect.width = endX-startX+1;
			refRect.height = endY-startY+1;
		}
		int r, c;
		Mat warpedImage;
		if (needMask && needAlpha)
		{
			warpedImage = Mat(newRow, newCol, CV_8UC4, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX,0));
		}
		else if (needMask && !needAlpha)
		{
			warpedImage = Mat(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
		}
		uchar *curWarpData = (uchar*)warpedImage.data;
		string filePath = _filePathList[curImgNo];
		Mat image = imread(filePath);
		uchar *curImgData = (uchar*)image.data;
		Mat_<double> invHomoMat = homoMat.inv();
		for (r = startY; r < endY; r ++)            
		{
			for (c = startX; c < endX; c ++)
			{
				int grayValueR, grayValueG, grayValueB;
				Point2d dstPt(c+mosaicRect.x,r+mosaicRect.y), srcPt(0,0);
				Utils::pointTransform(invHomoMat, dstPt, srcPt);
				int u = int(srcPt.x), v = int(srcPt.y);
				if (0+1 < u && width-2 > u && 0+1 < v && height-2 > v)
				{
					int grayValueR1 = 0, grayValueR2 = 0;
					int grayValueG1 = 0, grayValueG2 = 0;
					int grayValueB1 = 0, grayValueB2 = 0;

					//bilinear interpolation
					grayValueR1 = curImgData[3*(v*width+u)+0]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+0]*(srcPt.x-u);
					grayValueG1 = curImgData[3*(v*width+u)+1]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+1]*(srcPt.x-u);
					grayValueB1 = curImgData[3*(v*width+u)+2]*(1-(srcPt.x-u)) + curImgData[3*(v*width+u+1)+2]*(srcPt.x-u);

					grayValueR2 = curImgData[3*((v+1)*width+u)+0]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+0]*(srcPt.x-u);
					grayValueG2 = curImgData[3*((v+1)*width+u)+1]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+1]*(srcPt.x-u);
					grayValueB2 = curImgData[3*((v+1)*width+u)+2]*(1-(srcPt.x-u)) + curImgData[3*((v+1)*width+(u+1))+2]*(srcPt.x-u);

					grayValueR = grayValueR1*(1-(srcPt.y-v)) + grayValueR2*(srcPt.y-v);
					grayValueG = grayValueG1*(1-(srcPt.y-v)) + grayValueG2*(srcPt.y-v);
					grayValueB = grayValueB1*(1-(srcPt.y-v)) + grayValueB2*(srcPt.y-v);

					if (needMask)
					{
						if (!needAlpha)
						{
							curWarpData[3*(r*newCol+c)+0] = grayValueR;
							curWarpData[3*(r*newCol+c)+1] = grayValueG;
							curWarpData[3*(r*newCol+c)+2] = grayValueB;
						}
						else
						{
							warpedImage.at<Vec4b>(r,c)[3] = 255;   //! for alpha channel
							warpedImage.at<Vec4b>(r,c)[0] = grayValueR;  //! B
							warpedImage.at<Vec4b>(r,c)[1] = grayValueG;  //! G
							warpedImage.at<Vec4b>(r,c)[2] = grayValueB;  //! R
						}
					}
					//! set for the mosaic image
					mosaicData[3*(r*newCol+c)+0] = grayValueR;
					mosaicData[3*(r*newCol+c)+1] = grayValueG;
					mosaicData[3*(r*newCol+c)+2] = grayValueB;
				}
			}
		}
		if (!needMask)
		{
			continue;
		}
		char name[512];
		sprintf(name,"/Masks/warp%03d.png", curImgNo);
		string savePath = Utils::baseDir + string(name);
		imwrite(savePath, warpedImage);
	}

	if (0)
	{
		rectangle(stitchImage, Point2d(refRect.x,refRect.y), Point2d(refRect.x+refRect.width,refRect.y+refRect.height), Scalar(0,0,255), 5);
	}
	string filePath = Utils::baseDir + "/mosaic.png";
	imwrite(filePath, stitchImage);
	cout<<"-Completed!"<<endl;
}


double ImageAligner::CalWarpDeviation(vector<Point2d> pointSet1, vector<Point2d> pointSet2, Mat_<double> cvtMat, vector<double> weightList)
{
	unsigned i;
	int ptNum = pointSet1.size();
	if (weightList.size() == 0)
	{
		for (i = 0; i < ptNum; i ++)
		{
			weightList.push_back(1.0);
		}
	}
	double meanDist = 0, allWeight = 0;
//	vector<double> errorList;
	for (i = 0; i < ptNum; i ++)    //mean value
	{
		Point2d warpedPt;
		Utils::pointTransform(cvtMat, pointSet2[i], warpedPt);
		double dist = 0;
		dist = sqrt((warpedPt.x-pointSet1[i].x)*(warpedPt.x-pointSet1[i].x) + (warpedPt.y-pointSet1[i].y)*(warpedPt.y-pointSet1[i].y));
		meanDist += dist*weightList[i];
		allWeight += weightList[i];
	}
	meanDist /= allWeight;
	return meanDist;
}


void ImageAligner::recheckTopology(int sIndex, int eIndex)
{
	cout<<"-Recheck potential topology ..."<<endl;
	int addedPairs = 0;
	for (int i = sIndex; i <= eIndex; i ++)
	{
		int curNo = _visitOrder[i].imgNo;
		Mat_<double> affineMat = _alignModelList[i];
		//! calculating projecting centroid
		Quadra bar;
		bar.imgSize = _imgSizeList[curNo];
		Point2d centroid(bar.imgSize.width/2.0, bar.imgSize.height/2.0);	
		bar.centroid = Utils::pointTransform(affineMat, centroid);
		_projCoordSet.push_back(bar);

		vector<int> relatedNos = _matchNetList[curNo].relatedImgs;
		//! check with previously aligned images, except for its known neighbors
		for (int j = 0; j < sIndex; j ++)
		{
			bool isOldFriend = false;
			for (int k = 0; k < relatedNos.size(); k ++)
			{
				if (relatedNos[k] == j)
				{
					isOldFriend = true;
					break;
				}
			}
			if (isOldFriend)
			{
				continue;
			}
			int testNo = _visitOrder[j].imgNo;
			Quadra testObj = _projCoordSet[j];
			double threshold = 0.5*(max(bar.imgSize.width,bar.imgSize.height) + max(testObj.imgSize.width, testObj.imgSize.height));
			double dist = Utils::calPointDist(bar.centroid, testObj.centroid);
			if (dist > threshold*0.6)
			{
				continue;
			}
			vector<Point2d> PtSet1, PtSet2;
			bool yeah = _matcher->featureMatcher(curNo,testNo,PtSet1,PtSet2);
			if (yeah)
			{
				_similarityMat(testNo,curNo) = PtSet1.size();
				_similarityMat(curNo,testNo) = PtSet1.size();

				_matchNetList[curNo].relatedImgs.push_back(j);
				_matchNetList[curNo].PointSet.push_back(PtSet1);
				_matchNetList[testNo].relatedImgs.push_back(i);
				_matchNetList[testNo].PointSet.push_back(PtSet2);
				addedPairs ++;
			}
		}

	}
	cout<<"-Done! added "<<addedPairs<<" pairs."<<endl;
}


void ImageAligner::drawTopologyNet()
{
	vector<Point2d> coreLocations;
	int i, j, k;
	//! in the order of '_visitOrser'
	for (i = 0; i < _imgNum; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		double x0 = _imgSizeList[imgNo].width/2.0;
		double y0 = _imgSizeList[imgNo].height/2.0;
		Point2d centroid(x0, y0);
		Point2d warpPt;
		Utils::pointTransform(_alignModelList[i], centroid, warpPt);
		coreLocations.push_back(warpPt);
	}
	int minX = 999, minY = 999, maxX = 0, maxY = 0;
	for (i = 0; i < coreLocations.size(); i ++)
	{
		Point2d tmpPt = coreLocations[i];
		int x = int(fabs(tmpPt.x)+1)*(tmpPt.x < 0 ? -1 : 1);
		int y = int(fabs(tmpPt.y)+1)*(tmpPt.y < 0 ? -1 : 1);
		if (x > maxX)
			maxX = x;
		if (x < minX)
			minX = x;
		if (y > maxY)
			maxY = y;
		if (y < minY)
			minY = y;
	}
	double width = maxX - minX;
	double height = maxY - minY;
	int imageRange = 2000;             //maximum side_length
	int edgeRange = 60;
	double cvtScale = imageRange/min(height,width);
	int imageRow = height * cvtScale + edgeRange*2;   // add an edge space of 20 pixel
	int imageCol = width * cvtScale + edgeRange*2;
	Mat displayPlane(imageRow, imageCol, CV_8UC3, Scalar(255,255,255));

	CvFont font;
	double hScale = 1;
	double vScale = 1;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, hScale,vScale,0,1);      //定义标记字体
	vector<Point2i> dotPtList;
	//! in the order of '_visitOrser'
	for (i = 0; i < coreLocations.size(); i ++)
	{
		int c = int((coreLocations[i].x-minX) * cvtScale + 1) + edgeRange;
		int r = int((coreLocations[i].y-minY) * cvtScale + 1) + edgeRange;
		dotPtList.push_back(Point2i(c,r));
		circle(displayPlane, Point2i(c,r), 24, Scalar(255,0,0), -1);
//		circle(displayPlane, Point2i(c,r), 3, Scalar(255,255,0), -1);
		int imgNo = _visitOrder[i].imgNo;
		char text[100];
		sprintf(text,"%d", imgNo);
		Point2i dotPt(c+3, r+3);
//		cv::putText(displayPlane, text, dotPt, 2, 1, Scalar(0,0,0));
	}

	for (i = 0; i < _imgNum-1; i ++)         //draw all related lines
	{
		for (j = i+1; j < _imgNum; j ++)
		{
			int imgNo1 = _visitOrder[i].imgNo;
			int imgNo2 = _visitOrder[j].imgNo;
			int PtNum = _similarityMat(imgNo1,imgNo2);
			if (PtNum != 0)
			{
				Point2i startPt = dotPtList[i];
				Point2i endPt = dotPtList[j];				
				if (PtNum < 100 || 1)
				{
					line(displayPlane, startPt, endPt, Scalar(128,128,128), 2);
				}
				else
				{
					line(displayPlane, startPt, endPt, Scalar(0,255,0), 2);
				}
			}
		}
	}
	for (i = 1; i < _imgNum; i ++)        //! draw the related lines in MST
	{
		int refNo = _visitOrder[i].refNo;
		int refIndex = findVisitIndex(refNo);
		Point2i startPt = dotPtList[i];
		Point2i endPt = dotPtList[refIndex];
		line(displayPlane, startPt, endPt, Scalar(0,0,255), 3);
	}
	string savePath = Utils::baseDir + "/finalTopology.png";
	imwrite(savePath, displayPlane);
	cout<<"#the topology graph of images is saved!"<<endl;
}


void ImageAligner::labelGroupNodes()
{
	vector<Point2d> coreLocations;
	int i, j, k;
	//! in the order of '_visitOrser'
	for (i = 0; i < _imgNum; i ++)
	{
		int imgNo = _visitOrder[i].imgNo;
		double x0 = _imgSizeList[imgNo].width/2.0;
		double y0 = _imgSizeList[imgNo].height/2.0;
		Point2d centroid(x0, y0);
		Point2d warpPt;
		Utils::pointTransform(_alignModelList[i], centroid, warpPt);
		coreLocations.push_back(warpPt);
	}
	int minX = 999, minY = 999, maxX = 0, maxY = 0;
	for (i = 0; i < coreLocations.size(); i ++)
	{
		Point2d tmpPt = coreLocations[i];
		int x = int(fabs(tmpPt.x)+1)*(tmpPt.x < 0 ? -1 : 1);
		int y = int(fabs(tmpPt.y)+1)*(tmpPt.y < 0 ? -1 : 1);
		if (x > maxX)
			maxX = x;
		if (x < minX)
			minX = x;
		if (y > maxY)
			maxY = y;
		if (y < minY)
			minY = y;
	}
	double width = maxX - minX;
	double height = maxY - minY;
	int imageRange = 1000;             //maximum side_length
	int edgeRange = 50;
	double cvtScale = imageRange/min(height,width);
	int imageRow = height * cvtScale + edgeRange*2;   // add an edge space of 20 pixel
	int imageCol = width * cvtScale + edgeRange*2;
	Mat displayPlane(imageRow, imageCol, CV_8UC3, Scalar(255,255,255));

	CvFont font;
	double hScale = 1;
	double vScale = 1;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, hScale,vScale,0,1);      //定义标记字体
	vector<Point2i> dotPtList;
	//! label aligning group
	for (i = 0; i < _groupCusorList.size(); i ++)
	{
		int sIndex = 0, eIndex = 0;
		if (i != 0)
		{
			sIndex = _groupCusorList[i-1]+1;
			eIndex = _groupCusorList[i];
		}
		int r = rand()%255;
		int g = rand()%255;
		int b = rand()%255;
		for (j = sIndex; j <= eIndex; j ++)
		{
			int c = int((coreLocations[j].x-minX) * cvtScale + 1) + edgeRange;
			int r1 = int((coreLocations[j].y-minY) * cvtScale + 1) + edgeRange;
			dotPtList.push_back(Point2i(c,r1));
			circle(displayPlane, Point2i(c,r1), 25, Scalar(r,g,b), -1);
			int imgNo = _visitOrder[j].imgNo;
			char text[100];
			sprintf(text,"%d", imgNo);
			Point2i dotPt(c+3, r1+3);
			cv::putText(displayPlane, text, dotPt, 2, 1, Scalar(0,0,0));
		}
	}

	for (i = 0; i < _imgNum-1; i ++)         //draw all related lines
	{
		for (j = i+1; j < _imgNum; j ++)
		{
			int imgNo1 = _visitOrder[i].imgNo;
			int imgNo2 = _visitOrder[j].imgNo;
			int PtNum = _similarityMat(imgNo1,imgNo2);
			if (PtNum != 0)
			{
				Point2i startPt = dotPtList[i];
				Point2i endPt = dotPtList[j];				
				if (PtNum < 100)
				{
					line(displayPlane, startPt, endPt, Scalar(128,128,128), 1);
				}
				else
				{
					line(displayPlane, startPt, endPt, Scalar(0,255,0), 1);
				}
			}
		}
	}
	for (i = 1; i < _imgNum; i ++)        //! draw the related lines in MST
	{
		int refNo = _visitOrder[i].refNo;
		int refIndex = findVisitIndex(refNo);
		Point2i startPt = dotPtList[i];
		Point2i endPt = dotPtList[refIndex];
		line(displayPlane, startPt, endPt, Scalar(0,0,255), 2);
	}
	string savePath = Utils::baseDir + "/groupLabel.jpg";
	imwrite(savePath, displayPlane);
}


void ImageAligner::drawSimilarMatrix()
{
	int margin = 50;
	int imageRow = 800, imageCol = 800;
	int unitSize = 800/_imgNum;
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, hScale,vScale,0,1);      //define the label font
	Mat displayPlane(imageRow+2*margin, imageCol+2*margin, CV_8UC3, Scalar(0,0,0));
	for (int i = 0; i < _imgNum; i ++)
	{
		Point2i startPt_h(margin+i*unitSize, margin);        //draw vertical lines
		Point2i endPt_h(margin+i*unitSize, margin+unitSize*_imgNum);
		line(displayPlane, startPt_h, endPt_h, Scalar(128,128,128), 1);
		char text_h[100];
		sprintf(text_h,"%d", i+1);       
		Point2i dotPt1(startPt_h.x, startPt_h.y-10);         //draw horizontal labels(the dot is at left-bottom of the label)
		cv::putText(displayPlane, text_h, dotPt1, 1, 1, Scalar(255,255,255));

		Point2i startPt_v(margin, margin+i*unitSize);        //draw horizontal lines
		Point2i endPt_v(margin+unitSize*_imgNum, margin+i*unitSize);
		line(displayPlane, startPt_v, endPt_v, Scalar(128,128,128), 1);
		char text_v[100];
		sprintf(text_v,"%d", i+1);
		Point2i dotPt2(startPt_v.x-25, startPt_v.y+10);        //draw vertical labels
		cv::putText(displayPlane, text_v, dotPt2, 1, 1, Scalar(255,255,255));
	}
	int* numDataPtr = (int*)_similarityMat.data;
	int max_match_num = 0, min_match_num = 9999;
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{
			int num = numDataPtr[i*_imgNum+j];
			if (num > max_match_num)
			{
				max_match_num = num;
			}
			if (num < min_match_num)
			{
				min_match_num = num;
			}
		}
	}
	double grayScale = (255-64+1)*1.0 / (max_match_num-min_match_num+1);
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{
			Point2i leftTop, rigtBtm;
			leftTop.x = margin + i*unitSize;
			leftTop.y = margin + j*unitSize;
			rigtBtm.x = leftTop.x + unitSize;
			rigtBtm.y = leftTop.y + unitSize;

			int PtNum = _similarityMat(i,j);
			if (PtNum == 0)
			{
				continue;
			}
			int grayValue = 64 + int((PtNum-min_match_num) * grayScale);
			int r, c;
			for (c = leftTop.x; c < rigtBtm.x; c ++)
			{
				for (r = leftTop.y; r < rigtBtm.y; r ++)
				{
					//! down triangle
					displayPlane.at<Vec3b>(r,c)[0] = grayValue;
					displayPlane.at<Vec3b>(r,c)[1] = grayValue;
					displayPlane.at<Vec3b>(r,c)[2] = grayValue;
					//! up triangle
					displayPlane.at<Vec3b>(c,r)[0] = grayValue;
					displayPlane.at<Vec3b>(c,r)[1] = grayValue;
					displayPlane.at<Vec3b>(c,r)[2] = grayValue;
				}
			}
		}
	}
	string savePath = Utils::baseDir + "/similarTable.jpg";
	imwrite(savePath, displayPlane);
	cout<<"the similarity table of images is saved!"<<endl;
}


void ImageAligner::outputPrecise()
{
	cout<<"#Error analyzing ..."<<endl;
	double meanBias = 0;
	vector<double> deviations;
	int cnt = 0;
	for (int i = 1; i < _imgNum; i ++)
	{
		int curNo = _visitOrder[i].imgNo;
		Match_Net matchNet = _matchNetList[curNo];
		vector<int> relatedImgs = _matchNetList[curNo].relatedImgs;
		Mat_<double> homoMat1 = _alignModelList[i];
		Mat_<double> homoMat2;
		double curBias = 0;
		int cnt1 = 0;
		for (int j = 0; j < relatedImgs.size(); j ++)
		{
			int neigIndex = relatedImgs[j];
			if (neigIndex > i)
			{
				continue;
			}
			vector<Point2d> ptSet1, ptSet2;
			ptSet1 = matchNet.PointSet[j];         //! points on cur_image
			int neigNo = _visitOrder[neigIndex].imgNo;
			Match_Net neigMatchNet = _matchNetList[neigNo];
			for (int k = 0; k < neigMatchNet.relatedImgs.size(); k ++)
			{
				if (neigMatchNet.relatedImgs[k] == i)
				{
					ptSet2 = neigMatchNet.PointSet[k];
					break;
				}
			}
			homoMat2 = _alignModelList[neigIndex];
			//! using only one third of corresponding for optimization
			for (int t = 0; t < ptSet1.size(); t += 3)
			{
				Point2d mosaicPt1 = Utils::pointTransform(homoMat1, ptSet1[t]);
				Point2d mosaicPt2 = Utils::pointTransform(homoMat2, ptSet2[t]);
				double bias = sqrt((mosaicPt1.x-mosaicPt2.x)*(mosaicPt1.x-mosaicPt2.x)+(mosaicPt1.y-mosaicPt2.y)*(mosaicPt1.y-mosaicPt2.y));

				deviations.push_back(bias);
				meanBias += bias;
				cnt ++;
			}
		}
	}
	meanBias /= cnt;

	//! statistic the distribute of error
	string savePath = Utils::baseDir + "/precise.txt";
	ofstream fout(savePath, ios::out);
	if (!fout.is_open())
	{
		cout<<"Save path not exists!"<<endl;
		exit(0);
	}
	fout<<fixed<<setprecision(4);
	fout<<"RME : "<<meanBias<<endl;
	double minV = 0, maxV = 10, step = 0.1;
	int stepNum = int((maxV-minV)/step);
	for (int i = 1; i <= stepNum; i ++)
	{
		double rightV;
		rightV = minV + step*i;
		int accuNum = 0;
		for (int j = 0; j < deviations.size(); j ++)
		{
			double error = deviations[j];
			if (error <= rightV)
			{
				accuNum ++;
			}
		}
		double ratio = 1.0*accuNum/deviations.size();
		fout<<"0.0-"<<setw(3)<<rightV<<": "<<setw(4)<<ratio<<endl;
	}
	fout.close();
	cout<<"-Completed! with RME : "<<meanBias<<endl;
}


void ImageAligner::saveModelParams()
{
	cout<<"#Save aligning model params ..."<<endl;
	string savePath = Utils::baseDir + "/modelParmas.txt";
	ofstream fout(savePath, ios::out);
	if (!fout.is_open())
	{
		cout<<"Save path not exists!"<<endl;
		exit(0);
	}
	for (int i = 0; i < _imgNum; i ++)
	{
		int curImgNo = i;
		int curIndex = findVisitIndex(curImgNo);
		double* dataPtr = (double*)_alignModelList[curIndex].data;
		for (int j = 0; j < 8; j ++)
		{
			fout<<dataPtr[j]<<"  ";
		}
		fout<<endl;
	}
	cout<<"-Completed!"<<endl;
}


void ImageAligner::loadHomographies()
{
	cout<<"#Load aligning model params ..."<<endl;
	string filePath = Utils::baseDir + "/modelParmas.txt";
	ifstream fin(filePath, ios::in);
	if (!fin.is_open())
	{
		cout<<"File not exists!"<<endl;
		exit(0);
	}
	//! initializing
	for (int i = 0; i < _imgNum; i ++)
	{
		Mat_<double> homoMat = Mat::eye(3,3,CV_64FC1);
		_alignModelList.push_back(homoMat);
	}
	double u0 = 500.971, v0 = 323.486;
	Mat_<double> trans0 = (Mat_<double>(3,3) << 1, 0, -u0,
		                                       0, -1, 641-v0,
		                                       0, 0, 1);
	Mat_<double> trans1 = (Mat_<double>(3,3) << 1, 0, u0,
		                                       0, -1, -641+v0,
		                                       0, 0, 1);
	double kappa = 180;
	Mat_<double> Rz = (Mat_<double>(3,3) << cos(kappa), -sin(kappa), 0,
		                                    sin(kappa), cos(kappa), 0,
		                                        0,               0, 1);
	Mat_<double> t1 = (Mat_<double>(3,3) << 1, 0, -u0, 0, 1, -v0, 0, 0, 1);
	Mat_<double> t2 = (Mat_<double>(3,3) << 1, 0, u0, 0, 1, v0, 0, 0, 1);
	Mat_<double> bar = t2*Rz*t1;
	for (int i = 0; i < _imgNum; i ++)
	{
		Mat_<double> homoMat = Mat::eye(3,3,CV_64FC1);
		double* dataPtr = (double*)homoMat.data;
		for (int j = 0; j < 8; j ++)
		{
			double param = 0;
			fin >> param;
			dataPtr[j] = param;
		}
		int curImgNo = i;
		int curIndex = findVisitIndex(curImgNo);
		if ((i/31)%2 == 0)  //! rotate 180
		{
			_alignModelList[curIndex] = trans1*homoMat*trans0;
		}
		else
		{
			_alignModelList[curIndex] = trans1*homoMat*trans0;
		}
	}
	fin.close();
	cout<<"-Completed!"<<endl;
}