#include "featureMatch.h"

void PointMatcher::featureExtractor(bool extraPoints)
{
	int i, j;
	if (!extraPoints)
	{
		for (i = 0; i < _imgNum; i ++)
		{
			int imgIndex = i;
			char saveName[1024];
			sprintf(saveName, "Cache/keyPtfile/keys%d", imgIndex);
			string saveName_ = Utils::baseDir + string(saveName);
			_keysFileList.push_back(saveName_);
		}
		loadImgSizeList();
		return;
	}
	SurfFeatureDetector detector(400);
	SurfDescriptorExtractor extractor;
	for (i = 0; i < _imgNum; i ++)
	{
		string imgPath = _imgNameList[i];
		Mat image = imread(imgPath);
		Size imgSize(image.cols, image.rows);
		_imgSizeList.push_back(imgSize);

		vector<KeyPoint> keyPts;
		Mat descriptors;
		detector.detect(image, keyPts);
		extractor.compute(image, keyPts, descriptors);

		int imgIndex = i;
		char saveName[1024];
		sprintf(saveName, "Cache/keyPtfile/keys%d", imgIndex);
		string saveName_ = Utils::baseDir + string(saveName);
		_keysFileList.push_back(saveName_);
		savefeatures(keyPts, descriptors, saveName_);
		cout<<imgIndex<<" keyPoint file saved!"<<endl;
	}
	saveImgSizeList();
}


void PointMatcher::savefeatures(vector<KeyPoint> keyPts, Mat descriptors, string saveName)
{
	FILE *fout;
	fout = fopen(saveName.c_str(), "w");

	int PtNum = keyPts.size();
	fprintf(fout, "%d\n", PtNum);
	unsigned i, j;
//	int n0 = 0, n1 = 0, n2 = 0, n3 = 0;
	for (i = 0; i < PtNum; i ++)
	{
		Point2d point = keyPts[i].pt;
		int octave = keyPts[i].octave;
		fprintf(fout, "%.8lf %.8lf %d\n", point.x, point.y, octave);
		//if (octave == 0)
		//{
		//	n0 ++;
		//}
		//else if (octave == 1)
		//{
		//	n1 ++;
		//}
		//else if (octave == 2)
		//{
		//	n2 ++;
		//}
		//else
		//{
		//	n3 ++;
		//}
	}
	//int nn = n0+n1+n2+n3;
	//cout<<n0*1.0/nn<<" "<<n1*1.0/nn<<" "<<n2*1.0/nn<<" "<<n3*1.0/nn<<"  from "<<nn<<endl;
	for (i = 0; i < PtNum; i ++)        //write feature descriptor data
	{
		for (j = 0; j < featureDimension; j ++)
		{
			float temp = descriptors.at<float>(i,j);
			fprintf(fout, "%.8f ", temp);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}
 

void PointMatcher::readfeatures(int imgIndex, vector<Point2d> &keyPts, Mat &descriptors, double ratio)
{
	string fileName = _keysFileList[imgIndex];
	FILE *fin = fopen(fileName.c_str(), "r");
	int PtNum = 0;
	fscanf(fin, "%d", &PtNum);

	int step = int(1/ratio);
	int realNum = 0;
	unsigned i, j;
	for (i = 0; i < PtNum; i ++)
	{
		Point2d point;
		int octave = 0;
		fscanf(fin, "%lf%lf%d", &point.x, &point.y, &octave);
		if (i%step == 0)
		{
			keyPts.push_back(point);
			realNum++;
		}
	}
	descriptors = Mat(realNum, featureDimension, CV_32FC1, Scalar(0));
	int cnt = 0;
	for (i = 0; i < PtNum; i ++)        //write feature descriptor data
	{
		if (i%step != 0)
		{
			for (j = 0; j < featureDimension; j ++)
			{
				float temp;
				fscanf(fin, "%f", &temp);	
			}
		}
		else
		{
			for (j = 0; j < featureDimension; j ++)
			{
				float temp;
				fscanf(fin, "%f", &temp);	
				descriptors.at<float>(cnt,j) = temp;
			}
			cnt++;
		}
	}
	fclose(fin);
}


void PointMatcher::loadImgSizeList()
{
	string filePath = Utils::baseDir + "Cache/imgSizeList.txt";
	FILE *fin = fopen(filePath.c_str(), "r");
	if (fin == nullptr)
	{
		cout<<"No image size file!\n";
		return;
	}
	for (int i = 0; i < _imgNum; i ++)
	{
		int width, height;
		fscanf(fin, "%d  %d\n", &width, &height);
		_imgSizeList.push_back(Size(width,height));
	}
	fclose(fin);
}


void PointMatcher::saveImgSizeList()
{
	string savePath = Utils::baseDir + "Cache/imgSizeList.txt";
	FILE *fout = fopen(savePath.c_str(), "w");
	for (int i = 0; i < _imgNum; i ++)
	{
		int width = _imgSizeList[i].width, height = _imgSizeList[i].height;
		fprintf(fout, "%d  %d\n", width, height);
	}
	fclose(fout);
}


bool PointMatcher::featureMatcher(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
{
	pointSet1.clear();
	pointSet2.clear();
	vector<Point2d> keyPts1, keyPts2;
	Mat descriptors1, descriptors2;
	readfeatures(imgIndex1, keyPts1, descriptors1, 1.0);
	readfeatures(imgIndex2, keyPts2, descriptors2, 1.0);

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
		if (dist < 3.0)
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
//	cout<<"Image "<<imgIndex1<<" and image "<<imgIndex2<<" matched "<<pointSet1.size()<<" points"<<endl;

	////! chouxi
	//int step = max(1,int(pointSet1.size()/30));
	//vector<Point2d> set1, set2;
	//for (int i = 0; i < pointSet1.size(); i += step)
	//{
	//	set1.push_back(pointSet1[i]);
	//	set2.push_back(pointSet2[i]);
	//}
	//saveMatchPts(imgIndex1, imgIndex2, set1, set2);
	//drawMatches(imgIndex1, imgIndex2, set1, set2);
	saveMatchPts(imgIndex1, imgIndex2, pointSet1, pointSet2);
//	drawMatches(imgIndex1, imgIndex2, pointSet1, pointSet2);
	return true;
}


void PointMatcher::saveMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> pointSet1, vector<Point2d> pointSet2)
{
	bool exchanged = false;
	if (imgIndex1 > imgIndex2)          //set a consistent standard: smaller index in the left
	{
		int temp = imgIndex2;
		imgIndex2 = imgIndex1;
		imgIndex1 = temp;
		exchanged = true;
	}
	char saveName[1024];
	sprintf(saveName, "Cache/matchPtfile/match%d&%d.txt", imgIndex1, imgIndex2);
	string savePath = Utils::baseDir + string(saveName);
	FILE *fout = fopen(savePath.c_str(), "w");
	int PtNum = pointSet1.size();
	fprintf(fout, "%d\n", PtNum);
	if (!exchanged)
	{
		for (int i = 0; i < PtNum; i ++)
		{
			double x1 = pointSet1[i].x, y1 = pointSet1[i].y;
			double x2 = pointSet2[i].x, y2 = pointSet2[i].y;
			fprintf(fout, "%lf %lf %lf %lf\n", x1, y1, x2, y2);
		}
	}
	else
	{
		for (int i = 0; i < PtNum; i ++)
		{
			double x1 = pointSet1[i].x, y1 = pointSet1[i].y;
			double x2 = pointSet2[i].x, y2 = pointSet2[i].y;
			fprintf(fout, "%lf %lf %lf %lf\n", x2, y2, x1, y1);
		}
	}
	fclose(fout);
//	cout<<"Matched Points of image "<<imgIndex1<<" & image "<<imgIndex2<<" saved!"<<endl;
}


bool PointMatcher::loadMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
{
	bool exchanged = false;
	if (imgIndex1 > imgIndex2)          //set a consistent standard: smaller index in the left
	{
		int temp = imgIndex2;
		imgIndex2 = imgIndex1;
		imgIndex1 = temp;
		exchanged = true;
	}
	char fileName[1024];
	sprintf(fileName, "Cache/matchPtfile/match%d&%d.txt", imgIndex1, imgIndex2);
	string filePath = Utils::baseDir + string(fileName);
	FILE *fin = fopen(filePath.c_str(), "r");
	if (fin == nullptr)
	{
		cout<<"invalid matching file of image "<<imgIndex1<<" & image "<<imgIndex2<<endl;
		return false;
	}
	int PtNum = 0;
	fscanf(fin, "%d", &PtNum);
	if (!exchanged)
	{
		for (int i = 0; i < PtNum; i ++)
		{
			double x1, y1, x2, y2;
			fscanf(fin, "%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
			Point2d point1(x1,y1), point2(x2,y2);
			pointSet1.push_back(point1);
			pointSet2.push_back(point2);
		}
	}
	else
	{
		for (int i = 0; i < PtNum; i ++)
		{
			double x1, y1, x2, y2;
			fscanf(fin, "%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
			Point2d point1(x1,y1), point2(x2,y2);
			pointSet1.push_back(point2);
			pointSet2.push_back(point1);
		}
	}
	fclose(fin);
	cout<<"Loaded "<<pointSet1.size()<<" points between image "<<imgIndex1<<" and image "<<imgIndex2<<endl;
}


bool PointMatcher::tentativeMatcher(int imgIndex1, int imgIndex2)
{
	vector<Point2d> keyPts1, keyPts2;
	Mat descriptors1, descriptors2;
	readfeatures(imgIndex1, keyPts1, descriptors1, 0.3);
	readfeatures(imgIndex2, keyPts2, descriptors2, 0.3);

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
		double dist = knnmatches[i][0].distance;
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
	if (iniPts1.size() < 15)
	{
		return false;
	}
	Mat_<double> homoMat = findHomography(iniPts1, iniPts2, CV_RANSAC, 5.0);     //initial solution : from image2 to image1
	vector<Point2d> goodPts1, goodPts2;
	for (i = 0; i < iniPts1.size(); i ++)    //mean value
	{
		Point2d warpedPt;
		pointConvert(homoMat, iniPts2[i], warpedPt);
		double dist = 0;
		dist = sqrt((warpedPt.x-iniPts1[i].x)*(warpedPt.x-iniPts1[i].x) + (warpedPt.y-iniPts1[i].y)*(warpedPt.y-iniPts1[i].y));
		if (dist < 5.0)
		{
			goodPts1.push_back(iniPts1[i]);
			goodPts2.push_back(iniPts2[i]);
		}
	}
	if (goodPts1.size() < 5)
	{
		return false;
	}

	return true;
}


void PointMatcher::pointConvert(Mat_<double> homoMat, Point2d src, Point2d &dst)
{
	Mat_<double> srcX = (Mat_<double>(3,1)<< src.x, src.y, 1);
	Mat_<double> dstX = homoMat * srcX;
	dst = Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
}


void PointMatcher::drawMatches(int imgIndex1, int imgIndex2, vector<Point2d> pointSet1, vector<Point2d> pointSet2)
{
	int i, j;
	string fileName1 = _imgNameList[imgIndex1];
	string fileName2 = _imgNameList[imgIndex2];
	Mat image1 = imread(fileName1);
	Mat image2 = imread(fileName2);
	int w = 8;
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, hScale,vScale,0,1);      //定义标记字体
	for (i = 0; i < pointSet1.size(); i ++)
	{
		Point2d tempPt1 = pointSet1[i];
		circle(image1, tempPt1, 3, Scalar(0,0,255), -1);

		Point2d tempPt2 = pointSet2[i];
		circle(image2, tempPt2, 3, Scalar(0,0,255), -1);

/*		char text[100];
		sprintf(text,"%d", i);
		Point2d dotPt(3, 3);
		cv::putText(image1, text, tempPt1+dotPt, 2, 1, Scalar(0,0,0));
		cv::putText(image2, text, tempPt2+dotPt, 2, 1, Scalar(0,0,0));*/
		line(image1, tempPt1, tempPt2, Scalar(0,255,0), 1);
		line(image2, tempPt2, tempPt1, Scalar(0,255,0), 1);
	}
	char name1[512], name2[512];
	static int no = 0;
	sprintf(name1, "match%d_0.jpg", no);
	sprintf(name2, "match%d_1.jpg", no);
	no ++;
	string filePath1 = Utils::baseDir + "Cache/match_map/" + string(name1);
	string filePath2 = Utils::baseDir + "Cache/match_map/" + string(name2);
	imwrite(filePath1, image1);
	imwrite(filePath2, image2);
}