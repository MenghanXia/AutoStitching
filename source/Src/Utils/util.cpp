#include "util.h"

vector<string> Utils::get_filelist(string foldname)
{
	string mainPath = foldname;
	foldname += "/*.*";
	const char * mystr=foldname.c_str();
	vector<string> flist;
	string lineStr;
	vector<string> extendName;
	extendName.push_back("jpg");
	extendName.push_back("png");
	extendName.push_back("tif");
	extendName.push_back("JPG");
	extendName.push_back("bmp");

	HANDLE file;
	WIN32_FIND_DATA fileData;
	char line[1024];
	wchar_t fn[1000];
	mbstowcs(fn,mystr,999);
	file = FindFirstFile(fn, &fileData);
	FindNextFile(file, &fileData);
	while(FindNextFile(file, &fileData))
	{
		wcstombs(line,(const wchar_t*)fileData.cFileName,259);
		lineStr = line;
		for (int i = 0; i < 5; i ++)      //ÅÅ³ý·ÇÍ¼ÏñÎÄ¼þ
		{
			if (lineStr.find(extendName[i]) < 999)
			{
				lineStr = mainPath + "/" + lineStr;
				flist.push_back(lineStr);
				break;
			}
		}	
	}
	cout<<"loaded "<<flist.size()<<" files"<<endl;
	return flist;
}


Mat Utils::grayToPesudoColor(Mat grayMap)
{
	//! ºì 255, 0, 0   -> 255
	//! ³È 255, 127, 0 -> 204
	//! »Æ 255, 255, 0 -> 153
	//! ÂÌ 0, 255, 0   -> 102
	//! Çà 0, 255, 255 -> 51
	//! À¶ 0, 0, 255   -> 0
	int row = grayMap.rows, col = grayMap.cols;
	Mat colorMap(row, col, CV_8UC3, Scalar(0,0,0));
	uchar *dataPtr = (uchar*)grayMap.data;
	uchar *dataPtrT = (uchar*)colorMap.data;
	for (int i = 0; i < row; i ++)
	{
		for (int j = 0; j < col; j ++)
		{
			int pix = dataPtr[i*col+j];
			int B = 0, G = 0, R = 0;
			if (pix <= 51)
			{
				B = 255;
				G = pix*5;
				R = 0;
			}
			else if (pix <= 102)
			{
				pix-=51;
				B = 255-pix*5;
				G = 255;
				R = 0;
			}
			else if (pix <= 153)
			{
				pix-=102;
				B = 0;
				G = 255;
				R = pix*5;
			}
			else if (pix <= 204)
			{
				pix-=153;
				B = 0;
				G = 255-uchar(128.0*pix/51.0+0.5);
				R = 255;
			}
			else
			{
				pix-=204;
				B = 0;
				G = 127-uchar(127.0*pix/51.0+0.5);
				R = 255;
			}
			dataPtrT[3*(i*col+j)+0] = B;
			dataPtrT[3*(i*col+j)+1] = G;
			dataPtrT[3*(i*col+j)+2] = R;
			//colorMap.at<Vec3b>(i,j)[0] = B;
			//colorMap.at<Vec3b>(i,j)[1] = G;
			//colorMap.at<Vec3b>(i,j)[2] = R;
		}
	}
	return colorMap;
}


bool Utils::loadMatchPts(int imgIndex1, int imgIndex2, vector<Point2d> &pointSet1, vector<Point2d> &pointSet2)
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
//	cout<<"Loaded "<<pointSet1.size()<<" points between image "<<imgIndex1<<" and image "<<imgIndex2<<endl;
}


Mat_<double> Utils::buildCostGraph(const Mat_<int> &similarMat)
{
	int nodeNum = similarMat.rows;
	//! considering the precise and robustness, we take logarithm as the weight function
	Mat_<double> costGraph = Mat(nodeNum, nodeNum, CV_64FC1, Scalar(-1));
	for (int i = 0; i < nodeNum-1; i ++)
	{
		for (int j = i+1; j < nodeNum; j ++)
		{
			int num = similarMat(i,j);
			if (num == 0)
			{
				continue;
			}
			double cost = 6/log(num+50.0);
			costGraph(i,j) = cost;
			costGraph(j,i) = cost;
		}
	}
	return costGraph;
}


Point2d Utils::pointTransform(Mat_<double> homoMat, Point2d srcPt)
{
	Mat_<double> srcX = (Mat_<double>(3,1)<< srcPt.x, srcPt.y, 1);
	Mat_<double> dstX = homoMat * srcX;
	Point2d dstPt = Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
	return dstPt;
}


void Utils::pointTransform(Mat_<double> homoMat, Point2d srcPt, Point2d &dstPt)
{
	Mat_<double> srcX = (Mat_<double>(3,1)<< srcPt.x, srcPt.y, 1);
	Mat_<double> dstX = homoMat * srcX;
	dstPt = Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
}


void Utils::pointTransform(Mat_<double> homoMat, vector<Point2d> &pointSet)
{
	for (int i = 0; i < pointSet.size(); i ++)
	{
		Mat_<double> srcX = (Mat_<double>(3,1)<< pointSet[i].x, pointSet[i].y, 1);
		Mat_<double> dstX = homoMat * srcX;
		Point2d dstPt = Point2d(dstX(0)/dstX(2), dstX(1)/dstX(2));
		pointSet[i] = dstPt;
	}
}


double Utils::calPointDist(Point2d point1, Point2d point2)
{
	return sqrt((point1.x-point2.x)*(point1.x-point2.x) + (point1.y-point2.y)*(point1.y-point2.y));
}


double Utils::calVecDot(Point2d vec1, Point2d vec2)
{
	return vec1.x*vec2.x+vec1.y*vec2.y;
}