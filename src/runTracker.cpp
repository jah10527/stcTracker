#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "STCTracker.h"
/*****************************************************************************/
//command line
//1)input -b init.txt -v david.mpg, choose david video and give initial position
//2)input -v david.mpg, choose david but you should set initial position
//3)no any input, open camera and you should set initial position
/*****************************************************************************/
//-v david.mpg -b david_init.txt
#define STC 1

// Global variables
Rect box;
Rect STCbox;
bool drawing_box = false;
bool gotBB = false;
Rect boxRegion;//STC algorithm display context region 
bool fromfile =false;
string video;

static float TestTime=0.f;
int frameCount=1;

void readBB(char* file)	// get tracking box from file
{
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

void print_help(void)
{
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			//printf("-b%d\n",i);
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			//printf("-v%d\n",i);
			if (argc>i)
			{
				video = string(argv[i+1]);
				capture.open(video);				
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}
	}
}

// bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

int main(int argc, char * argv[])
{
	VideoCapture capture;

	// Read options
	read_options(argc, argv,capture);
	Mat frame;
	Mat first;
	capture.open("David/img/%04d.jpg"); 
	// Init camera
	if (!capture.isOpened())
	{
		cout<<"capture device failed to open!"<<endl;
		return -1;
	}

    box.x = 165;
    box.y = 93;
    box.width = 51;
    box.height = 54;
    gotBB = true; 

    capture >> frame;  

#ifdef STC
	STCbox = box;
	//STC initialization
	STCTracker stcTracker;
	stcTracker.init(frame, STCbox,boxRegion);
#endif
	// Run-time
	int frameCount = 1;
	
	while (1)
	{
		capture >> frame;
		if (frame.empty())
			break;
		
#ifdef STC
		float t = (float)cvGetTickCount();
		// tracking
		stcTracker.tracking(frame, STCbox,boxRegion,frameCount);	
		t = (float)cvGetTickCount() - t;
		cout << "cost time: " << t / ((float)cvGetTickFrequency()*1000000.) <<" s,";
		cout<<cvRound(((float)cvGetTickFrequency()*1000000.)/t)<<"FPS"<<endl;
		cout<<"object size:="<<STCbox.width<<"*"<<STCbox.height<<endl;
#endif
		frameCount++;

		// show the result
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(15, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(100, 0, 255), 3);

#ifdef STC
		putText(frame, "      STC", Point(80, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		rectangle(frame, STCbox, Scalar(0, 0, 255), 3);
		rectangle(frame, boxRegion, Scalar(255, 200, 255), 3);
#endif

		imshow("Tracker.jpg", frame);
    waitKey(5);
	}
	return 0;
}