#pragma once

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

class STCTracker
{
public:
	STCTracker();
	~STCTracker();
	void init(const Mat &frame, const Rect box,Rect &boxRegion);	
	void tracking(const Mat &frame, Rect &trackBox,Rect &boxRegion,int FrameNum);

private:
	void createHammingWin();
	void complexOperation(const Mat &src1, const Mat &src2, Mat &dst, int flag = 0);
	void getCxtPriorPosteriorModel(const Mat &image);
	void learnSTCModel(Mat &image);

private:
	float sigma;			// scale parameter (variance)
	float alpha, ialpha;	// scale parameter
	float beta;				// shape parameter
	float rho;				// learning parameter
	float scale, scale_curr;			//	scale ratio
	float lambda;			//	scale learning parameter
	int num;				//	the number of frames for updating the scale
	vector<float> maxValue;
	Point center;			//	the object position
	Rect cxtRegion;			// context region
	int padding;
	Scalar average;
	
	Point point;
	double  maxVal;
	
	Mat cxtPriorPro;		// prior probability
	Mat cxtPosteriorPro;	// posterior probability
	Mat STModel;			// conditional probability
	Mat STCModel;			// spatio-temporal context model
	Mat hammingWin;			// Hamming window
	Mat gray;
	Mat context;
	Mat conditionalFourier;
	Mat postFourier;
	Mat priorFourier;
	Mat STCModelFourier;
	Mat confidenceMap;
	Mat planes1[2], planes2[2], planes3[2];
	Mat zeros;
};