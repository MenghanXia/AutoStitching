#include "featureMatch.h"
#include "alignment.h"
#include "Utils/util.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <windows.h>

using namespace std;

void DOMCheckbyRigid();

void main()
{
	clock_t start_time, end_time;
	start_time = clock();

	string imagePath = Utils::baseDir + "Images/classic1";
	vector<string> imgPathList = Utils::get_filelist(imagePath);
//	for (int i = 0; i < imgPathList.size(); i ++)
//	{
//		if ((i%31)%2 != 0)
//		{
//			continue;
//		}
//		Mat img = imread(imgPathList[i]);
////		cout<<imgPathList[i]<<endl;
//		char name[512];
//		sprintf(name, "E:/bar/img%02d_%02d.jpg",i/31+1,i%31+1);
////		cout<<name<<endl;
//		string fileName = name;
//		imwrite(fileName, img);
//	}
//	return;
	ImageAligner imgAligner(imgPathList);
	int referNo = -1;
//	imgAligner.imageStitcherbySolos(referNo);
	imgAligner.imageStitcherbyGroup(referNo);
	end_time=clock();
	cout<<"All done! consumed "<<(end_time-start_time)/CLOCKS_PER_SEC<<" seconds! "<<endl;
}


void DOMCheckbyRigid()
{
	vector<Point2d> pts1, pts2;
	Utils::loadMatchPts(0, 1, pts1, pts2);
	Mat_<double> A = Mat(2*21, 4, CV_64FC1, Scalar(0));
	Mat_<double> L = Mat(2*21, 1, CV_64FC1, Scalar(0));
	for (int k = 0; k < pts1.size(); k ++)
	{
		int rn = k;
		Point2d pt1 = pts2[k], pt2 = pts1[k];
		A(2*rn,0)     = pt1.x; A(2*rn,1)   = -pt1.y; A(2*rn,2) = 1;
		A(2*rn+1,0) = pt1.y; A(2*rn+1,1) = pt1.x; A(2*rn+1,3) = 1;
		L(2*rn)   = pt2.x;
		L(2*rn+1) = pt2.y;
	}
	Mat_<double> X = (A.t()*A).inv()*(A.t()*L);
	Mat_<double> simiModel = (Mat_<double>(3,3) << X(0)  , -X(1), X(2),
		X(1), X(0), X(3),
		0,      0,     1);
	FILE *fp = fopen("E:/adf.txt", "w");
	for (int k = 0; k < pts2.size(); k ++)
	{
		Point2d tar = Utils::pointTransform(simiModel, pts2[k]);
		fprintf(fp, "%lf  %lf\n", tar.x, tar.y);
	}
	fclose(fp);
	return;
}