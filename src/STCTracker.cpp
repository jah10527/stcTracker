#include "STCTracker.h"
#include <sys/time.h>
#include <iostream>

STCTracker::STCTracker(){}
STCTracker::~STCTracker(){}


float InvSqrt(float x)
{
 float xhalf = 0.5f*x;
 int i = *(int*)&x; 		// get bits for floating VALUE
 i = 0x5f375a86- (i>>1); 	// gives initial guess y0
 x = *(float*)&i; 			// convert bits BACK to float
 x = x*(1.5f-xhalf*x*x); 	// Newton step, repeating increases accuracy
 //x = x*(1.5f-xhalf*x*x); 	// Newton step, repeating increases accuracy
 //x = x*(1.5f-xhalf*x*x); 	// Newton step, repeating increases accuracy
 return x;
}


/************ Create a Hamming window ********************/
void STCTracker::createHammingWin()
{
	int i, j;
	float* data = hammingWin.ptr<float>(0);
	for (i = 0; i < hammingWin.rows; i++)
	{
		for (j = 0; j < hammingWin.cols; j++)
		{
			*data++ = (0.54 - 0.46 * cos( 2 * CV_PI * i / hammingWin.rows ))*(0.54 - 0.46 * cos( 2 * CV_PI * j / hammingWin.cols  ));
		}
	}
}

/************ Define two complex-value operation *****************/
void STCTracker::complexOperation(const Mat &src1, const Mat &src2, Mat &dst, int flag)
{
	CV_Assert(src1.size == src2.size);
	CV_Assert(src1.channels() == 2);

	const float* data1;
	const float* data2;
	float* data3;
	float a, b, c, d, t;
	int i, j;
/*	data1 = src1.ptr<float>(0);
	data2 = src2.ptr<float>(0);
	data3 = dst.ptr<float>(0);*/
	data1 = (const float*)src1.data;
	data2 = (const float*)src2.data;
	data3 = (float*)dst.data;
	j = src1.rows * src1.cols;
	for (i = 0; i < j; i++)
	{
//		for (j = 0; j < src1.cols; j++)
//		{
			a = *(data1++);
			b = *(data1++);
			c = *(data2++);
			d = *(data2++);

			if (flag)
			{
				// division: (a+bj) / (c+dj)
				t = 1/(c * c + d * d + 0.000001);
				*(data3++) = (a * c + b * d) * t;
				*(data3++) = (b * c - a * d) * t;
			}
			else
			{
				// multiplication: (a+bj) * (c+dj)
				*(data3++) = a * c - b * d;
				*(data3++) = b * c + a * d;
			}
//		}
	}
}

/************ Get context prior and posterior probability ***********/
void STCTracker::getCxtPriorPosteriorModel(const Mat &image)
{
	//cout<<"cxtPriorPro "<<cxtPriorPro.rows<<" "<<cxtPriorPro.cols<<endl;
	//cout<<"img"<<image.rows<<" "<<image.cols<<endl;
	CV_Assert(image.size == cxtPriorPro.size);

	float sum_prior(0), sum_post(0);
	float* data;
	float* data1;
	float x, y, dist;
	int i, j, isigma;
	data = cxtPriorPro.ptr<float>(0);
	data1 = cxtPosteriorPro.ptr<float>(0);
	isigma = 0.5f/(sigma * sigma);
	for (i = 0; i < cxtRegion.height; i++)
	{
		y = i + cxtRegion.y;
		for (j = 0; j < cxtRegion.width; j++)
		{
			x = j + cxtRegion.x;
//			y = i + cxtRegion.y;
			dist = (center.x - x) * (center.x - x) + (center.y - y) * (center.y - y);

			// equation (5) in the paper
			*data = exp(- dist * isigma);
			sum_prior += *data++;

			// equation (6) in the paper
			*data1 = exp(- pow(dist * ialpha, beta));
			sum_post += *data1++;
		}
	}
	cxtPriorPro.convertTo(cxtPriorPro, -1, 1.0/sum_prior);
	cxtPriorPro = cxtPriorPro.mul(image);
	cxtPosteriorPro.convertTo(cxtPosteriorPro, -1, 1.0/sum_post);
}

/************ Learn Spatio-Temporal Context Model ***********/
void STCTracker::learnSTCModel(Mat &image)
{
	// step 1: Get context prior and posterior probability
	getCxtPriorPosteriorModel(image);
	
	// step 2-1: Execute 2D DFT for prior probability
/*	Mat priorFourier;
	Mat planes1[] = {cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_32F)};
    merge(planes1, 2, priorFourier);*/
	merge(planes1, 2, priorFourier);
	dft(priorFourier, priorFourier);

	// step 2-2: Execute 2D DFT for posterior probability
/*	Mat postFourier;
	Mat planes2[] = {cxtPosteriorPro, Mat::zeros(cxtPosteriorPro.size(), CV_32F)};*/
    merge(planes3, 2, postFourier);
	dft(postFourier, postFourier);

	// step 3: Calculate the division
	complexOperation(postFourier, priorFourier, conditionalFourier, 1);

	// step 4: Execute 2D inverse DFT for conditional probability and we obtain STCModel
	dft(conditionalFourier, STModel, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	// step 5: Use the learned spatial context model to update spatio-temporal context model
	addWeighted(STCModel, 1.0 - rho, STModel, rho, 0.0, STCModel);
}

/************ Initialize the hyper parameters and models ***********/
void STCTracker::init(const Mat &frame, const Rect box,Rect &boxRegion)
{
	// initial some parameters
	padding=1;
	num=5;         		//num consecutive frames
	alpha = 2.25;		//parameter \alpha in Eq.(6)
	ialpha = InvSqrt(alpha);
	beta = 1;		 	//Eq.(6)
	rho = 0.075;		//learning parameter \rho in Eq.(12)
	lambda=0.25;
	sigma = 0.5 * (box.width + box.height);//sigma init
	scale=1.0;
	sigma=sigma*scale;

	// the object position
	center.x = box.x + 0.5 * box.width;
	center.y = box.y + 0.5 * box.height;

	// the context region
	cxtRegion.width = (1+padding) * box.width;
	cxtRegion.height = (1+padding) * box.height;
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	cxtRegion.y = center.y - cxtRegion.height * 0.5;	
	cxtRegion &= Rect(0, 0, frame.cols, frame.rows);
	boxRegion=cxtRegion;//output box region

	// the prior, posterior and conditional probability and spatio-temporal context model
	cxtPriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_32FC1);
	cxtPosteriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_32FC1);
	STModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_32FC1);
	STCModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_32FC1);
	conditionalFourier.create(cxtRegion.height, cxtRegion.width, CV_32FC2);
	postFourier.create(cxtRegion.height, cxtRegion.width, CV_32FC2);
	zeros = Mat::zeros(cxtPriorPro.size(), CV_32F);
	planes1[0] = cxtPriorPro;
	planes1[1] = zeros;
	planes2[0] = STCModel;
	planes2[1] = zeros;
	planes3[0] = cxtPosteriorPro;
	planes3[1] = zeros;

	// create a Hamming window
	hammingWin = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_32FC1);
	createHammingWin();

	cvtColor(frame, gray, CV_RGB2GRAY);

	// normalized by subtracting the average intensity of that region
	average = mean(gray(cxtRegion));
	gray(cxtRegion).convertTo(context, CV_32FC1, 1.0, - average[0]);

	// multiplies a Hamming window to reduce the frequency effect of image boundary
	context = context.mul(hammingWin);

	// learn Spatio-Temporal context model from first frame
	learnSTCModel(context);
}

/******** STCTracker: calculate the confidence map and find the max position *******/
void STCTracker::tracking(const Mat &frame, Rect &trackBox,Rect &boxRegion,int FrameNum)
{
	cvtColor(frame, gray, CV_RGB2GRAY);

	// normalized by subtracting the average intensity of that region
	average = mean(gray(cxtRegion));
	gray(cxtRegion).convertTo(context, CV_32FC1, 1.0, - average[0]);

	// multiplies a Hamming window to reduce the frequency effect of image boundary
	context = context.mul(hammingWin);

	// step 1: Get context prior probability
	//cout<<"context "<<context.rows<<" "<<context.cols<<endl;
	getCxtPriorPosteriorModel(context);


	// step 2-1: Execute 2D DFT for prior probability
/*	Mat priorFourier;
	Mat planes1[] = {cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_32F)};
    merge(planes1, 2, priorFourier);*/
	merge(planes1, 2, priorFourier);
	dft(priorFourier, priorFourier);

	// step 2-2: Execute 2D DFT for conditional probability
/*	Mat STCModelFourier;
	Mat planes2[] = {STCModel, Mat::zeros(STCModel.size(), CV_32F)};*/
    merge(planes2, 2, STCModelFourier);
	dft(STCModelFourier, STCModelFourier);

	// step 3: Calculate the multiplication
	complexOperation(STCModelFourier, priorFourier, postFourier, 0);

	// step 4: Execute 2D inverse DFT for posterior probability namely confidence map
	dft(postFourier, confidenceMap, DFT_INVERSE | DFT_REAL_OUTPUT| DFT_SCALE);
	
	// step 5: Find the max position
	minMaxLoc(confidenceMap, 0, &maxVal, 0, &point);
	maxValue.push_back(maxVal);

		/***********update scale by Eq.(15)**********/
//	if (FrameNum%(num+2)==0)
	{   
		scale_curr=0.0;

		for (int k=0;k<num;k++)
		{
//			scale_curr+=sqrt(maxValue[FrameNum-k-2]/maxValue[FrameNum-k-3]);
			scale_curr+=InvSqrt(maxValue[FrameNum-k-3]/maxValue[FrameNum-k-2]);
		}

		scale=(1-lambda)*scale+lambda*(scale_curr/num);

		sigma=sigma*scale;

	}
	// step 6-1: update center, trackBox and context region
	center.x = cxtRegion.x + point.x;
	center.y = cxtRegion.y + point.y;
	trackBox.x = center.x - 0.5 * trackBox.width;
	trackBox.y = center.y - 0.5 * trackBox.height;
	trackBox &= Rect(0, 0, frame.cols, frame.rows);
	//boundary
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	if (cxtRegion.x<0)
	{
		cxtRegion.x=0;
	}
	cxtRegion.y = center.y - cxtRegion.height * 0.5;
	if (cxtRegion.y<0)
	{
		cxtRegion.y=0;
	}
	if (cxtRegion.x+cxtRegion.width>frame.cols)
	{
		cxtRegion.x=frame.cols-cxtRegion.width;
	}
	if (cxtRegion.y+cxtRegion.height>frame.rows)
	{
		cxtRegion.y=frame.rows-cxtRegion.height;
	}
	
	//cout<<"cxtRegionXY"<<cxtRegion.x<<" "<<cxtRegion.y<<endl;
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;
	//cout<<"frame"<<frame.rows<<" "<<frame.cols<<endl;

	//cxtRegion &= Rect(0, 0, frame.cols, frame.rows);
	//cout<<"cxtRegionXY"<<cxtRegion.x<<" "<<cxtRegion.y<<endl;
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;

	
	boxRegion=cxtRegion;
	// step 7: learn Spatio-Temporal context model from this frame for tracking next frame
	average = mean(gray(cxtRegion));
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;

	gray(cxtRegion).convertTo(context, CV_32FC1, 1.0, - average[0]);
	
	//cout<<"hamm"<<hammingWin.rows<<" "<<hammingWin.cols<<endl;

	context = context.mul(hammingWin);
	learnSTCModel(context);

}