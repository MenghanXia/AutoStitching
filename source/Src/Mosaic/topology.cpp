#include "topology.h"
#define BKGRNDPIX 0

Mat_<double> TopoFinder::findTopology(bool shallLoad, bool isInOrder)
{
	_isInOrder = isInOrder;
	if (shallLoad)
	{
		//! load similarity matrix from disk
		return loadSimilarityMat();
	}
	loadKeyFiles();
	clock_t start_time, end_time;
	start_time = clock();
	cout<<"Building similarity table ...\n";
	Mat_<int> similarMat = detectSimilarityByGuiding();
	//		Mat_<int> similarMat = detectSimilarityOnGlobal();
	end_time = clock();
	_totalTime = 1000*(end_time-start_time)/CLOCKS_PER_SEC;   //! ms
	//! write out 
	string savePath = Utils::baseDir + "topoInfor.txt";
	ofstream fout;
	fout.open(savePath.c_str(), ios::out);
	fout<<_shotNum<<"  in  "<<_attempNum<<endl;
	fout<<_matchTime<<"  of   "<<_totalTime<<endl;
	fout.close();
	saveSimilarityMat(similarMat);
	return similarMat;
}


Mat_<int> TopoFinder::detectSimilarityOnGlobal()
{
	Mat_<int> similarMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{		
			_attempNum ++;
			vector<Point2d> pointSet1, pointSet2;
//			Utils::loadMatchPts(i,j,pointSet1,pointSet2);
			if (featureMatcher(i,j,pointSet1,pointSet2))
			{
				similarMat(i,j) = pointSet1.size();
				similarMat(j,i) = pointSet1.size();
				_shotNum ++;
			}
		}
	}
	return similarMat;
}


Mat_<int> TopoFinder::detectSimilarityByGuiding()
{
	bool isTimeConsecutive = _isInOrder;
	_similarityMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	_attempMap = Mat::eye(_imgNum, _imgNum, CV_16UC1);
	//! find and match main chain, meanwhile edit the similarity table
	if (isTimeConsecutive)
	{
		buildMainChain();
	}
	else
	{
		searchMainChain();
	}
	//! solve the aligning models of each image
	Mat_<double> identMatrix = Mat::eye(3,3,CV_64FC1);
	_affineMatList.push_back(identMatrix);
	Quadra bar;
	bar.imgSize = _Ptmatcher->_imgSizeList[_visitOrder0[0].imgNo];
	bar.centroid = Point2d(bar.imgSize.width/2, bar.imgSize.height/2);
	_projCoordSet.push_back(bar);
	cout<<"Detecting potential overlaps ..."<<endl;
	for (int i = 1; i < _visitOrder0.size(); i ++)
	{
		cout<<"No."<<i<<" finding ...";
		int curNo = _visitOrder0[i].imgNo;
		int refNo = _visitOrder0[i].refNo;
		int refIndex = findNodeIndex(refNo);
		vector<Point2d> pointSet1, pointSet2;
		clock_t st, et;
		st = clock();
		Utils::loadMatchPts(refNo, curNo, pointSet1, pointSet2);
		et = clock();
		_matchTime -= (et-st);
	    Utils::pointTransform(_affineMatList[refIndex], pointSet1);
		//! perform initial alignment
		Mat_<double> affineMat = findFastAffine(pointSet1, pointSet2);
		_affineMatList.push_back(affineMat);
		//! record centroid of current image
		Quadra bar;
		bar.imgSize = _Ptmatcher->_imgSizeList[curNo];
		Point2d centroid(bar.imgSize.width/2.0, bar.imgSize.height/2.0);	
		bar.centroid = Utils::pointTransform(affineMat, centroid);
		_projCoordSet.push_back(bar);

		//! detect potential overlaps
		//! 1) recalculate aligning model; 2) modify centroid; 3) modify similarity table
		detectPotentialOverlap(i, pointSet1, pointSet2);
		cout<<"-->end!"<<endl;
	}
	drawTopoNet();
//	drawTreeLevel();
//	TsaveMosaicImage();
	return _similarityMat;
}


Mat_<double> TopoFinder::getGuidingTable()
{
	cout<<"Initializing ..."<<endl;
	Mat_<double> guidingTable = Mat(_imgNum, _imgNum, CV_64FC1, Scalar(0));
	Mat_<int> simiMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{		
			int num = calSimilarNum(i,j);
			simiMat(i,j) = num;
			simiMat(j,i) = num;
			double cost = 6/log(num+50.0);
			if (num == 0)
			{
				cost = -1;
			}
			guidingTable(i,j) = cost;
			guidingTable(j,i) = cost;
		}
	}
	saveSimilarityMat(simiMat);
	cout<<"Done!"<<endl;
	return guidingTable;
}


Mat_<double> TopoFinder::getGuidingTableP()
{
	cout<<"Initializing ..."<<endl;
	Mat_<double> guidingTable = Mat(_imgNum, _imgNum, CV_64FC1, Scalar(0));
	Mat_<int> simiMat = Mat(_imgNum, _imgNum, CV_32SC1, Scalar(0));
	int step = max(1,(_imgNum-1)*_imgNum/20);      //! 10%
	int n = 0;
	for (int i = 0; i < _imgNum-1; i ++)
	{
		for (int j = i+1; j < _imgNum; j ++)
		{		
			int num = calSimilarNum(i,j);
			simiMat(i,j) = num;
			simiMat(j,i) = num;
			double cost = 6/log(num+50.0);
			if (num == 0)
			{
				cost = -1;
			}
			guidingTable(i,j) = cost;
			guidingTable(j,i) = cost;
			n ++;
			if (n%step == 0)
			{
				cout<<10*n/step<<"% ";
			}
		}
	}
	cout<<endl;
	saveSimilarityMat(simiMat);
	cout<<"Done!"<<endl;
	return guidingTable;
}


void TopoFinder::buildMainChain()
{
	cout<<"Building main chain according to the time consecutive order ..."<<endl;
	int refeNo = _imgNum/2;
	TreeNode bar(refeNo,-1,0);
	_visitOrder0.push_back(bar);
	int offset = 1;
	while (1)
	{
		int no1 = refeNo-offset, no2 = refeNo+offset;
		TreeNode bar1(no1,no1+1,0), bar2(no2,no2-1,0);
		_visitOrder0.push_back(bar1);
		_visitOrder0.push_back(bar2);
		vector<Point2d> pointSet1, pointSet2;
		if (Load_Matches)
		{
			Utils::loadMatchPts(no1, no1+1, pointSet1, pointSet2);
		}
		else
		{
			clock_t st, et;
			st = clock();
			if (!featureMatcher(no1, no1+1, pointSet1, pointSet2))
			{
				cout<<no1<<"&"<<no1+1<<"Time consecutive sequence break up!"<<endl;
				exit(0);
			}
			et = clock();
			_matchTime += (et-st);
		}
		_similarityMat(no1, no1+1) = pointSet1.size();
		_similarityMat(no1+1, no1) = pointSet1.size();
		_attempMap(no1, no1+1) = 1;
		_attempMap(no1+1, no1) = 1;
		if (Load_Matches)
		{
			Utils::loadMatchPts(no2, no2-1, pointSet1, pointSet2);
		}
		else
		{
			clock_t st, et;
			st = clock();
			if (!featureMatcher(no2, no2-1, pointSet1, pointSet2))
			{
				cout<<no2<<"&"<<no2-1<<"Time consecutive sequence break up!"<<endl;
				exit(0);
			}
			et = clock();
			_matchTime += (et-st);
		}
		_similarityMat(no2, no2-1) = pointSet1.size();
		_similarityMat(no2-1, no2) = pointSet1.size();
		_attempMap(no2, no2-1) = 1;
		_attempMap(no2-1, no2) = 1;
		offset ++;
		if (no1 == 0 || no2 == _imgNum-1)
		{
			if (no1 > 0)
			{
				_visitOrder0.push_back(TreeNode(no1-1,no1,0));
				if (Load_Matches)
				{
					Utils::loadMatchPts(no1-1, no1, pointSet1, pointSet2);
				}
				else
				{
					clock_t st, et;
					st = clock();
					if (!featureMatcher(no1-1, no1, pointSet1, pointSet2))
					{
						cout<<no1-1<<"&"<<no1<<"Time consecutive sequence break up!"<<endl;
						exit(0);
					}
					et = clock();
					_matchTime += (et-st);
				}
				_similarityMat(no1-1, no1) = pointSet1.size();
				_similarityMat(no1, no1-1) = pointSet1.size();
				_attempMap(no1-1, no1) = 1;
				_attempMap(no1, no1-1) = 1;
			}
			else if (no2 < _imgNum-1)
			{
				_visitOrder0.push_back(TreeNode(no2+1,no2,0));
				if (Load_Matches)
				{
					Utils::loadMatchPts(no2+1, no2, pointSet1, pointSet2);
				}
				else
				{
					clock_t st, et;
					st = clock();
					if (!featureMatcher(no2+1, no2, pointSet1, pointSet2))
					{
						cout<<no2<<"&"<<no2+1<<"Time consecutive sequence break up!"<<endl;
						exit(0);
					}
					et = clock();
					_matchTime += (et-st);
				}
				_similarityMat(no2+1, no2) = pointSet1.size();
				_similarityMat(no2, no2+1) = pointSet1.size();
				_attempMap(no2+1, no2) = 1;
				_attempMap(no2, no2+1) = 1;
			}
			break;
		}
	}
	_attempNum = _imgNum-1;
	_shotNum = _imgNum-1;
	cout<<"Completed!"<<endl;
}


void TopoFinder::searchMainChain()
{
	//! overlapping probability of image pairs
	Mat_<double> guidingTable = getGuidingTableP();
	int iter = 0, maxIter = max(int(_imgNum*0.2), 20);
	while (1)
	{
		cout<<"Searching main chain ... (attempt: "<<++iter<<")"<<endl;
		Mat_<int> imgPairs= Graph::extractMSTree(guidingTable);
		int pairNo = 0;
		for (pairNo = 0; pairNo < imgPairs.rows; pairNo ++)
		{
			int no1 = imgPairs(pairNo,0), no2 = imgPairs(pairNo,1);
			//! avoiding repeating matching which is done in last iteration
			if (_attempMap(no1,no2) != 0)
			{
				continue;
			}
			_attempMap(no1,no2) = 1;
			_attempMap(no2,no1) = 1;
			vector<Point2d> pointSet1, pointSet2;
			_attempNum ++;

			clock_t st, et;
			st = clock();
			bool yeah = featureMatcher(no1,no2,pointSet1,pointSet2);
			et = clock();
			_matchTime += (et-st);
			if (yeah)
			{
				_shotNum ++;
				_similarityMat(no1,no2) = pointSet1.size();
				_similarityMat(no2,no1) = pointSet1.size();
				guidingTable(no1,no2) = 0.0;
				guidingTable(no2,no1) = 0.0;
			}
			else
			{
				//! matching failed : cost as infinite
				guidingTable(no1,no2) = 999;
				guidingTable(no2,no1) = 999;
				if (iter == maxIter)
				{
					cout<<"Poor image sequence! exit out."<<endl;
					exit(0);
				}
				cout<<no1<<" Linking "<<no2<<" failed! < built: "<<pairNo+1<<" edges."<<endl;
				break;
			}
		}
		if (pairNo == _imgNum-1)
		{
			break;
		}
	}
	cout<<"Succeed!"<<endl;
	Mat_<double> costGraph = Utils::buildCostGraph(_similarityMat);
	_visitOrder0 = Graph::FloydForPath(costGraph);
}


void TopoFinder::detectPotentialOverlap(int curIndex, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
{
	int curRefNo = _visitOrder0[curIndex].refNo;
	int curNo = _visitOrder0[curIndex].imgNo;
	Point2d iniPos = _projCoordSet[curIndex].centroid;
	int width = _projCoordSet[curIndex].imgSize.width;
	int height = _projCoordSet[curIndex].imgSize.height;
	//! accelerate : build a KD-tree for all centroids and retrieve
	bool isGot = false;
	for (int i = 0; i < _projCoordSet.size(); i ++)
	{
		int testNo = _visitOrder0[i].imgNo;
		if (_attempMap(curNo,testNo))
		{
			continue;
		}
		Quadra testObj = _projCoordSet[i];
		double threshold = 0.5*(max(width,height) + max(testObj.imgSize.width, testObj.imgSize.height));
		double dist = Utils::calPointDist(iniPos, testObj.centroid);
		if (dist > threshold*0.8)
		{
			continue;
		}
		_attempNum ++;
		vector<Point2d> newPtSet1, newPtSet2;
		//! for debug test
		if (Load_Matches)
		{
			if (Utils::loadMatchPts(testNo,curNo,newPtSet1,newPtSet2))
			{
				_similarityMat(testNo,curNo) = newPtSet1.size();
				_similarityMat(curNo,testNo) = newPtSet1.size();
				Utils::pointTransform(_affineMatList[i], newPtSet1);
				for (int t = 0; t < newPtSet1.size(); t ++)
				{
					pointSet1.push_back(newPtSet1[t]);
					pointSet2.push_back(newPtSet2[t]);
				}
				_shotNum ++;
				isGot = true;
			}
		}
		else
		{
			if (dist < threshold*0.8)
			{
				clock_t st, et;
				st = clock();
				bool yeah = featureMatcher(testNo,curNo,newPtSet1,newPtSet2);
				et = clock();
				_matchTime += (et-st);
				if (yeah)
				{
					_similarityMat(testNo,curNo) = newPtSet1.size();
					_similarityMat(curNo,testNo) = newPtSet1.size();
					Utils::pointTransform(_affineMatList[i], newPtSet1);
					for (int t = 0; t < newPtSet1.size(); t ++)
					{
						pointSet1.push_back(newPtSet1[t]);
						pointSet2.push_back(newPtSet2[t]);
					}
					_shotNum ++;
					isGot = true;
				}
			}
			//else
			//{
			//	if (_Ptmatcher->tentativeMatcher(testNo,curNo))
			//	{
			//		clock_t st, et;
			//		st = clock();
			//		bool yeah = _Ptmatcher->featureMatcher(testNo,curNo,newPtSet1,newPtSet2);
			//		et = clock();
			//		_matchTime += (et-st);
			//		_similarityMat(testNo,curNo) = newPtSet1.size();
			//		_similarityMat(curNo,testNo) = newPtSet1.size();
			//		Utils::pointTransform(_affineMatList[i], newPtSet1);
			//		for (int t = 0; t < newPtSet1.size(); t ++)
			//		{
			//			pointSet1.push_back(newPtSet1[t]);
			//			pointSet2.push_back(newPtSet2[t]);
			//		}
			//		_shotNum ++;
			//		isGot = true;
			//	}		
			//}
		}
	}
	if (!isGot)
	{
		return;
	}
	//! modify the affine model parameter
	_affineMatList[curIndex] = findFastAffine(pointSet1, pointSet2);
	//! modify the centroid
	_projCoordSet[curIndex].centroid = Utils::pointTransform(_affineMatList[curIndex], Point2d(width/2,height/2));
}


int TopoFinder::findNodeIndex(int imgNo)
{
	int imgIndex = 0;
	for (int i = 0; i < _imgNum; i ++)
	{
		if (_visitOrder0[i].imgNo == imgNo)
		{
			imgIndex = i;
		}
	}
	return imgIndex;
}


Mat_<int> TopoFinder::loadSimilarityMat()
{
	string filePath = Utils::baseDir + "Cache/similarityMat.txt";
	ifstream fin;
	fin.open(filePath.c_str(), ios::in);
	if (!fin.is_open())
	{
		cout<<"File not found!\n";
		exit(0);
	}
	int nodeNum = _imgNum;
	Mat_<double> similarityMat = Mat(nodeNum, nodeNum, CV_64FC1, Scalar(0));
	for (int i = 0; i < nodeNum; i ++)
	{
		//! avoid to read repeated data
		//int offset = sizeof(int)*i;
		//fin.seekg(offset, ios::cur);
		for (int j = 0; j < nodeNum; j ++)
		{
			int Ptnum = 0;
			fin>>Ptnum;
			similarityMat(i,j) = Ptnum;
		}
	}
	fin.close();

	return similarityMat;
}


void TopoFinder::saveSimilarityMat(const Mat_<int> &similarityMat)
{
	string savePath = Utils::baseDir + "Cache/similarityMat.txt";
	ofstream fout;
	fout.open(savePath.c_str(), ios::out);
	if (!fout.is_open())
	{
		cout<<"Path not exists!\n";
		exit(0);
	}
	int nodeNum = similarityMat.cols;
//	fout<<fixed<<setprecision(2);
	for (int i = 0; i < nodeNum; i ++)
	{
		for (int j = 0; j < nodeNum; j ++)
		{
			int Ptnum = similarityMat(i,j);
			fout<<setw(4)<<Ptnum<<" ";
		}
		fout<<endl;
	}
	fout.close();
}


Mat_<double> TopoFinder::findFastAffine(vector<Point2d> pointSet1, vector<Point2d> pointSet2)
{
	int step = max(1, int(pointSet1.size()/500));
	int pointNum = pointSet1.size()/step;
	Mat_<double> affineMat(3, 3, CV_64FC1);
	Mat A(2*pointNum, 6, CV_64FC1, Scalar(0));
	Mat L(2*pointNum, 1, CV_64FC1);
	for (int i = 0; i < pointNum; i ++)
	{
		double x1 = pointSet1[i*step].x, y1 = pointSet1[i*step].y;
		double x2 = pointSet2[i*step].x, y2 = pointSet2[i*step].y;
		A.at<double>(i*2,0) = x2; A.at<double>(i*2,1) = y2; A.at<double>(i*2,2) = 1; 
		A.at<double>(i*2+1,3) = x2; A.at<double>(i*2+1,4) = y2; A.at<double>(i*2+1,5) = 1;
		L.at<double>(i*2,0) = x1;
		L.at<double>(i*2+1,0) = y1;
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	affineMat(0,0) = X(0); affineMat(0,1) = X(1); affineMat(0,2) = X(2);
	affineMat(1,0) = X(3); affineMat(1,1) = X(4); affineMat(1,2) = X(5);
	affineMat(2,0) = 0; affineMat(2,1) = 0; affineMat(2,2) = 1;
	double var = 0;
	//Utils::pointTransform(affineMat, pointSet2);
	//for (int i = 0; i < pointNum; i ++)
	//{

	//	double bias = (pointSet1[i].x-pointSet2[i].x)*(pointSet1[i].x-pointSet2[i].x) + 
 //                     (pointSet1[i].y-pointSet2[i].y)*(pointSet1[i].y-pointSet2[i].y);
	//	var += bias;
	//}
	//var /= (pointNum-6);
	return affineMat;
}


void TopoFinder::loadKeyFiles()
{
	cout<<"Loading key files ..."<<endl;
	for (int i = 0; i < _imgNum; i ++)
	{
		int imgIndex = i;
		char filePath[1024];
		sprintf(filePath, "Cache/keyPtfile/keys%d", imgIndex);
		string filePath_ = Utils::baseDir + string(filePath);
		cout<<"key "<<i<<endl;
		Keys bar;
		vector<int> subIndexList;    //! features of the target ocatve

		FILE *fin = fopen(filePath_.c_str(), "r");
		int PtNum = 0;
		fscanf(fin, "%d", &PtNum);
		for (int j = 0; j < PtNum; j ++)
		{
			Point2d point;
			int octave = 0;
			fscanf(fin, "%lf%lf%d", &point.x, &point.y, &octave);
			bar.pts.push_back(point);
			if (octave == Target_Octave)
			{
				subIndexList.push_back(j);
			}
		}
		if (subIndexList.size() == 0)       //! special case
		{
			//cout<<"Waring : the feature subset of image "<<i<<" is empty!"<<endl;
			for (int j = 0; j < PtNum; j ++)
			{
				Point2d point;
				int octave = 0;
				fscanf(fin, "%lf%lf%d", &point.x, &point.y, &octave);
				bar.pts.push_back(point);
				int Gesus = max(0,Target_Octave-1);
				if (octave == Gesus)
				{
					subIndexList.push_back(j);
				}
			}
		}
		Mat descriptors = Mat(PtNum, 64, CV_32FC1, Scalar(0));
		int cnt = 0;
		for (int j = 0; j < PtNum; j ++)        //write feature descriptor data
		{
			for (int k = 0; k < 64; k ++)
			{
				float temp;
				fscanf(fin, "%f", &temp);
				descriptors.at<float>(j,k) = temp;
			}
		}
		bar.descriptors = descriptors;
		fclose(fin);

		_keyList.push_back(bar);
		_subKeyIndexList.push_back(subIndexList);
	}
	cout<<"Completed!"<<endl;
}


bool TopoFinder::featureMatcher(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
{
	pointSet1.clear();
	pointSet2.clear();
	vector<Point2d> keyPts1, keyPts2;
	keyPts1 = _keyList[imgIndex1].pts;
	keyPts2 = _keyList[imgIndex2].pts;
	Mat descriptors1, descriptors2;
	descriptors1 = _keyList[imgIndex1].descriptors;
	descriptors2 = _keyList[imgIndex2].descriptors;

	// Matching descriptor vectors using FLANN matcher
	vector<DMatch> m_Matches;
	FlannBasedMatcher matcher; 
	vector<vector<DMatch>> knnmatches;
	int num1 = keyPts1.size(), num2 = keyPts2.size();
	int kn = min(min(num1, num2), 5);
	matcher.knnMatch(descriptors1, descriptors2, knnmatches, kn);   
	int i, j;
	double minimaDsit = 99999;
	for (i = 0; i < knnmatches.size(); i ++)
	{
		double dist = knnmatches[i][1].distance;
		if (dist < minimaDsit)
		{
			minimaDsit = dist;
		}
	}
	double fitedThreshold = minimaDsit * 5;
	int keypointsize = knnmatches.size();
	for (i = 0; i < keypointsize; i ++)
	{  
		const DMatch nearDist1 = knnmatches[i][0];
		const DMatch nearDist2 = knnmatches[i][1];
		double distanceRatio = nearDist1.distance / nearDist2.distance;
		if (nearDist1.distance < fitedThreshold && distanceRatio < 0.7)
		{
			m_Matches.push_back(nearDist1);
		}
	}

	vector<Point2d> iniPts1, iniPts2;
	for (i = 0; i < m_Matches.size(); i ++)   //get initial match pairs
	{
		int queryIndex = m_Matches[i].queryIdx;
		int trainIndex = m_Matches[i].trainIdx;
		Point2d tempPt1 = keyPts1[queryIndex];
		Point2d tempPt2 = keyPts2[trainIndex];
		iniPts1.push_back(tempPt1);
		iniPts2.push_back(tempPt2);
	}
	if (iniPts1.size() < 10)
	{
		return false;
	}

	vector<uchar> status;
	//! utilize epi-polar geometry constraints to delete missing matches
	Mat Fmatrix = findFundamentalMat(iniPts1, iniPts2, CV_RANSAC, 1.5, 0.99, status);
	for (i = 0; i < status.size(); i ++)
	{
		if (status[i] == 1)
		{
			pointSet1.push_back(iniPts1[i]);
			pointSet2.push_back(iniPts2[i]);
		}
	}
	if (pointSet1.size() < 10)
	{
		return false;
	}

	Mat homoMat = findHomography(iniPts2, iniPts1, CV_RANSAC, 2.5);    //! Pt1 = homoMat*Pt2
	vector<Point2d> goodPts1, goodPts2;
	for (i = 0; i < pointSet1.size(); i ++)    //mean value
	{
		Point2d warpedPt;
		Utils::pointTransform(homoMat, pointSet2[i], warpedPt);
		double dist = 0;
		dist = sqrt((warpedPt.x-pointSet1[i].x)*(warpedPt.x-pointSet1[i].x) + (warpedPt.y-pointSet1[i].y)*(warpedPt.y-pointSet1[i].y));
		if (dist < 2.0)
		{
			goodPts1.push_back(pointSet1[i]);
			goodPts2.push_back(pointSet2[i]);
		}
	}
	pointSet1 = goodPts1;
	pointSet2 = goodPts2;
	if (pointSet1.size() < 10)    //! modify as 20
	{
		return false;
	}
	cout<<"Image "<<imgIndex1<<" and image "<<imgIndex2<<" matched "<<pointSet1.size()<<" points"<<endl;
	_Ptmatcher->saveMatchPts(imgIndex1, imgIndex2, pointSet1, pointSet2);
//	_Ptmatcher->drawMatches(imgIndex1, imgIndex2, pointSet1, pointSet2);
	return true;
}


int TopoFinder::calSimilarNum(int imgIndex1, int imgIndex2)
{
	vector<Point2d> orgKeyPts1 = _keyList[imgIndex1].pts;
	vector<Point2d> orgKeyPts2 = _keyList[imgIndex2].pts;
	Mat orgDescriptors1, orgDescriptors2;
	orgDescriptors1 = _keyList[imgIndex1].descriptors;
	orgDescriptors2 = _keyList[imgIndex2].descriptors;

	vector<Point2d> keyPts1, keyPts2;
	Mat descriptors1, descriptors2;

	vector<int> subSet1 = _subKeyIndexList[imgIndex1], subSet2 = _subKeyIndexList[imgIndex2];
	int realNum1 = subSet1.size(), realNum2 = subSet2.size();
	//! sample subset 1
	for (int i = 0; i < realNum1; i ++)
	{
		int no = subSet1[i];
		keyPts1.push_back(orgKeyPts1[no]);
	}
	descriptors1 = Mat(realNum1, 64, CV_32FC1);
	for (int i = 0; i < realNum1; i ++)
	{
		int no = subSet1[i];
		orgDescriptors1.row(no).copyTo(descriptors1.row(i));
	}
	//! sample subset 2
	for (int i = 0; i < realNum2; i ++)
	{
		int no = subSet2[i];
		keyPts2.push_back(orgKeyPts2[no]);
	}
	descriptors2 = Mat(realNum2, 64, CV_32FC1);
	for (int i = 0; i < realNum2; i ++)
	{
		int no = subSet2[i];
		orgDescriptors2.row(no).copyTo(descriptors2.row(i));
	}
	// Matching descriptor vectors using FLANN matcher
	vector<DMatch> m_Matches;
	FlannBasedMatcher matcher; 
	vector<vector<DMatch>> knnmatches;
	int num1 = keyPts1.size(), num2 = keyPts2.size();
	int kn = min(min(num1, num2), 5);
	matcher.knnMatch(descriptors1, descriptors2, knnmatches, kn);   
	double minimaDsit = 99999;
	for (int i = 0; i < knnmatches.size(); i ++)
	{
		double dist = knnmatches[i][0].distance;
		if (dist < minimaDsit)
		{
			minimaDsit = dist;
		}
	}
	double fitedThreshold = minimaDsit * 5;
	int keypointsize = knnmatches.size();
	for (int i = 0; i < keypointsize; i ++)
	{  
		const DMatch nearDist1 = knnmatches[i][0];
		const DMatch nearDist2 = knnmatches[i][1];
		double distanceRatio = nearDist1.distance / nearDist2.distance;
		if (nearDist1.distance < fitedThreshold && distanceRatio < 0.7)
		{
			m_Matches.push_back(nearDist1);
		}
	}
	int num = m_Matches.size();
	return num;
}


//! ========= temperate funcs ===========
void TopoFinder::drawTopoNet()
{
	int i, j, k;
	int minX = 999, minY = 999, maxX = 0, maxY = 0;
	for (i = 0; i < _projCoordSet.size(); i ++)
	{
		Point2d tmpPt = _projCoordSet[i].centroid;
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
	for (i = 0; i < _visitOrder0.size(); i ++)
	{
		Point2d point = _projCoordSet[i].centroid;
		int c = int((point.x-minX) * cvtScale + 1) + edgeRange;
		int r = int((point.y-minY) * cvtScale + 1) + edgeRange;
		dotPtList.push_back(Point2i(c,r));
		circle(displayPlane, Point2i(c,r), 24, Scalar(255,0,0), -1);
		//circle(displayPlane, Point2i(c,r), 2, Scalar(255,255,0), -1);
		int imgNo = _visitOrder0[i].imgNo;
		char text[100];
		sprintf(text,"%d", imgNo);
		Point2i dotPt(c+3, r+3);
//		cv::putText(displayPlane, text, dotPt, 2, 1, Scalar(0,0,0));
	}
	for (i = 0; i < _visitOrder0.size()-1; i ++)         //draw all related lines
	{
		for (j = i+1; j < _visitOrder0.size(); j ++)
		{
			int nodeNo1 = _visitOrder0[i].imgNo;
			int nodeNo2 = _visitOrder0[j].imgNo;
			int PtNum = _similarityMat(nodeNo1,nodeNo2);
			if (PtNum > 0)
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
/*	for (i = 1; i < _visitOrder0.size(); i ++)        //draw the related lines in MST
	{
		int refNo = _visitOrder0[i].refNo;
		int refIndex = findNodeIndex(refNo);
		Point2i startPt = dotPtList[i];
		Point2i endPt = dotPtList[refIndex];
		line(displayPlane, startPt, endPt, Scalar(0,0,255), 3);
	}*/
	string savePath = Utils::baseDir + "/topoGraph.png";
	imwrite(savePath, displayPlane);
	cout<<"The topology graph of images is saved!"<<endl;
}


Rect TopoFinder::TsetImageSize()
{
	vector<Point2d> marginPtList;
	int i, j;
	for (i = 0; i < _visitOrder0.size(); i ++)
	{
		Mat_<double> homoMat = _affineMatList[i];
		int curImgNo = _visitOrder0[i].imgNo;
		Size imgSize = _Ptmatcher->_imgSizeList[curImgNo];
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

	Rect mosaicRect;
	mosaicRect.x = minX; mosaicRect.y = minY;
	mosaicRect.width = maxX-minX+1; mosaicRect.height = maxY-minY+1;
	return mosaicRect;
}


void TopoFinder::TsaveMosaicImage()
{
	bool shallEach = false;
	Rect mosaicRect = TsetImageSize();
	int newRow = mosaicRect.height, newCol = mosaicRect.width;
	int i, j;
	Rect newImgRect;
	Mat stitchImage(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
	for (i = 0; i < _visitOrder0.size(); i ++)
	{
		int curImgNo = _visitOrder0[i].imgNo;
		cout<<"Warping Image: "<<curImgNo<<"..."<<endl;
		Mat_<double> homoMat = _affineMatList[i];
		Size imgSize = _Ptmatcher->_imgSizeList[curImgNo];
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
		Mat warpedImage(newRow, newCol, CV_8UC3, Scalar(BKGRNDPIX,BKGRNDPIX,BKGRNDPIX));
		string filePath = _Ptmatcher->_imgNameList[curImgNo];
		Mat image = imread(filePath);
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
					int grayValueR1 = BKGRNDPIX, grayValueR2 = BKGRNDPIX;
					int grayValueG1 = BKGRNDPIX, grayValueG2 = BKGRNDPIX;
					int grayValueB1 = BKGRNDPIX, grayValueB2 = BKGRNDPIX;
					//bilinear interpolation
					grayValueR1 = (image.at<Vec3b>(v,u)[0]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v,u+1)[0]) * (srcPt.x-u);
					grayValueG1 = (image.at<Vec3b>(v,u)[1]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v,u+1)[1]) * (srcPt.x-u);
					grayValueB1 = (image.at<Vec3b>(v,u)[2]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v,u+1)[2]) * (srcPt.x-u);

					grayValueR2 = (image.at<Vec3b>(v+1,u)[0]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v+1,u+1)[0]) * (srcPt.x-u);
					grayValueG2 = (image.at<Vec3b>(v+1,u)[1]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v+1,u+1)[1]) * (srcPt.x-u);
					grayValueB2 = (image.at<Vec3b>(v+1,u)[2]) * (1 - (srcPt.x-u)) + (image.at<Vec3b>(v+1,u+1)[2]) * (srcPt.x-u);

					grayValueR = grayValueR1*(1 - (srcPt.y-v)) + grayValueR2*(srcPt.y-v);
					grayValueG = grayValueG1*(1 - (srcPt.y-v)) + grayValueG2*(srcPt.y-v);
					grayValueB = grayValueB1*(1 - (srcPt.y-v)) + grayValueB2*(srcPt.y-v);

					warpedImage.at<Vec3b>(r,c)[0] = grayValueR;
					warpedImage.at<Vec3b>(r,c)[1] = grayValueG;
					warpedImage.at<Vec3b>(r,c)[2] = grayValueB;

					stitchImage.at<Vec3b>(r,c)[0] = grayValueR;
					stitchImage.at<Vec3b>(r,c)[1] = grayValueG;
					stitchImage.at<Vec3b>(r,c)[2] = grayValueB;
				}
			}
		}
		if (!shallEach)
		{
			continue;
		}
		char name[512];
		sprintf(name,"/Masks/warp%d.png", curImgNo);
		string savePath = Utils::baseDir + string(name);
		imwrite(savePath, warpedImage);
	}
	string filePath = Utils::baseDir + "/topoMosaic.png";
	imwrite(filePath, stitchImage);
}


void TopoFinder::drawTreeLevel()
{
	Mat_<double> costGraph = Utils::buildCostGraph(_similarityMat);
	vector<TreeNode> newVisits = Graph::FloydForPath(costGraph);
	vector<Point2d> coreLocations;
	int i, j, k;
	//! update the order of visit list
	for (i = 0; i < newVisits.size(); i ++)
	{
		int imgNo = newVisits[i].imgNo;
		int orgIndex = findNodeIndex(imgNo);
		coreLocations.push_back(_projCoordSet[orgIndex].centroid);
	}
	//! label levels of spinning tree
	vector<int> groupCusorList;
	groupCusorList.push_back(0);
	for (int i = 1; i < newVisits.size(); i ++)
	{
		if (newVisits[i].level != newVisits[i-1].level)
		{
			groupCusorList.push_back(i);
		}
	}
	int groupNum = groupCusorList.size();
	if (groupCusorList[groupNum-1] < _imgNum-1)
	{
		groupCusorList.push_back(_imgNum-1);
	}
	int minX = 999, minY = 999, maxX = 0, maxY = 0;
	for (i = 0; i < _projCoordSet.size(); i ++)
	{
		Point2d tmpPt = _projCoordSet[i].centroid;
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
	int edgeRange = 30;
	double cvtScale = imageRange/min(height,width);
	int imageRow = height * cvtScale + edgeRange*2;   // add an edge space of 20 pixel
	int imageCol = width * cvtScale + edgeRange*2;
	Mat displayPlane(imageRow, imageCol, CV_8UC3, Scalar(255,255,255));
	//! label aligning group
	vector<Point2i> dotPtList;
	for (i = 0; i < groupCusorList.size(); i ++)
	{
		int sIndex = 0, eIndex = 0;
		if (i != 0)
		{
			sIndex = groupCusorList[i-1]+1;
			eIndex = groupCusorList[i];
		}
		int r = rand()%255;
		int g = rand()%255;
		int b = rand()%255;
		for (j = sIndex; j <= eIndex; j ++)
		{
			Point2d point = coreLocations[j];
			int c = int((point.x-minX) * cvtScale + 1) + edgeRange;
			int r1 = int((point.y-minY) * cvtScale + 1) + edgeRange;
			dotPtList.push_back(Point2i(c,r1));
			circle(displayPlane, Point2i(c,r1), 25, Scalar(r,g,b), -1);
		}
	}
	//for (i = 0; i < _imgNum-1; i ++)         //draw all related lines
	//{
	//	for (j = i+1; j < _imgNum; j ++)
	//	{
	//		int imgNo1 = _visitOrder0[i].imgNo;
	//		int imgNo2 = _visitOrder0[j].imgNo;
	//		int PtNum = _similarityMat(imgNo1,imgNo2);
	//		if (PtNum != 0)
	//		{
	//			Point2i startPt = dotPtList[i];
	//			Point2i endPt = dotPtList[j];				
	//			if (PtNum < 100)
	//			{
	//				line(displayPlane, startPt, endPt, Scalar(128,128,128), 1);
	//			}
	//			else
	//			{
	//				line(displayPlane, startPt, endPt, Scalar(0,255,0), 1);
	//			}
	//		}
	//	}
	//}
	for (i = 1; i < newVisits.size(); i ++)        //draw the related lines in MST
	{
		int refNo = newVisits[i].refNo;
		int refIndex = findNodeIndex(refNo);
		Point2i startPt = dotPtList[i];
		Point2d refNoPos = _projCoordSet[refIndex].centroid;
		int c = int((refNoPos.x-minX) * cvtScale + 1) + edgeRange;
		int r1 = int((refNoPos.y-minY) * cvtScale + 1) + edgeRange;
		Point2i endPt(c,r1);
		line(displayPlane, startPt, endPt, Scalar(0,0,255), 2);
	}
	string savePath = Utils::baseDir + "/spinningTree.jpg";
	imwrite(savePath, displayPlane);
}