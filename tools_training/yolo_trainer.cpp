#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

class LabelEntry {
	public:
	 //string type;		// Describes the type of object: e.g. 'Car', 'Van', 'Truck',
                     		// 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     		// 'Misc' or 'DontCare'
	 int type;      // index of type

	 float bbox_x;	// 2D bounding box of object in the image (0-based index):
         float bbox_y;	// contains left, top, right, bottom pixel coordinates
	 float bbox_width;  // NOTE that these are all relative to image width and height!!
	 float bbox_height;             		
	
	 LabelEntry() {
	    //type = "DontCare";
            type = 0;
	    bbox_x = 0.0f;
	    bbox_y = 0.0f;
	    bbox_width = 0.0f;
	    bbox_height = 0.0f;
	}
};	
	
std::ostream& operator<<(std::ostream& out, const LabelEntry& obj) {
	return out << obj.type << " " 
		   << obj.bbox_x << " " 
		   << obj.bbox_y << " " 
	   	   << obj.bbox_width << " "
		   << obj.bbox_height;
}

class Region {
	public:
	 LabelEntry label;
	 Rect cropRect;
	 bool selected;

	 Region() {
		cropRect = Rect(0,0,0,0);
		selected = false;
	 }
};

// training video vars
string videoFilename;
bool frameAdvance = true;
int startingFrame = 0;

// training region vars
Mat src,img;
Rect activeCropRect(0,0,0,0);
Point P1(0,0);
Point P2(0,0);
vector<string> classes;
list<Region> regions;
Region *activeRegionPtr = NULL;
int offsetX = 0;
int offsetY = 0;
string selectedClass;

// training data vars
string trainingFilenamePrefix;
string trainingImagePath;
string trainingLabelPath;
bool trainingDataExportEnable = false;
long frameCounter = 0;

const char* winName="Crop Image";
bool leftclicked=false;
bool rightclicked=false;
char imgName[15];


// Here, 'DontCare' labels denote regions in which objects have not been labeled,
// for example because they have been too far away from the laser scanner. To
// prevent such objects from being counted as false positives our evaluation
// script will ignore objects detected in don't care regions of the test set.
// You can use the don't care labels in the training set to avoid that your object
// detector is harvesting hard negatives from those areas, in case you consider
// non-object regions from the training images as negative examples.

void showRegions(){
	
    //int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1.0f;
    int thickness = 2;
    
    img=src.clone();

    // show active cropRect (green)
    rectangle(img, activeCropRect, Scalar(0,255,0), 1, 8, 0 );
    
    // show current regions
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
	// show region type
	Point pt;
	pt.x = (*list_iter).cropRect.tl().x;
	pt.y = (*list_iter).cropRect.tl().y - 10;
	string typeStr = classes[(*list_iter).label.type];
	putText(img,typeStr,pt,fontFace,fontScale,cvScalar(200,200,250),thickness,8, false);

	// show region rect
	if ((*list_iter).selected)
    		rectangle(img, (*list_iter).cropRect, Scalar(0,0,255), 2, 8, 0 );
	else
    		rectangle(img, (*list_iter).cropRect, Scalar(0,255,255), 2, 8, 0 );

    }

    // show training data state
    Point pt2(img.cols/2 - 20,img.rows - 20);
    if (trainingDataExportEnable)
	putText(img,"TRAINING ON",pt2,fontFace,1,cvScalar(0,0,255), 2 ,8, false);
    else
	putText(img,"TRAINING OFF",pt2,fontFace,1,cvScalar(0,255,0), 2 ,8, false);

    imshow(winName,img);

}

void checkRegions(int x, int y) {

    activeRegionPtr = NULL;
    offsetX = 0;
    offsetY = 0;
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
	// if the current region contains the mouse click point...
	if ((*list_iter).cropRect.contains(Point(x,y))) 
	{
		// toggle selected flag on region
		(*list_iter).selected = !((*list_iter).selected);
		if ((*list_iter).selected)
		{
			// save pointer to active region
			activeRegionPtr = &(*list_iter);

			// record offsets from point to region rect origin
			offsetX = x - (*list_iter).cropRect.x;
			offsetY = y - (*list_iter).cropRect.y;
		}
		else
			// reset active region pointer
			activeRegionPtr = NULL;
	}
	else
		(*list_iter).selected = false;
    }

}

void moveRegion(Point p) {
    if (activeRegionPtr != NULL) 
    {
	    // update bounding box pos
	    (*activeRegionPtr).cropRect.x = p.x - offsetX;
	    (*activeRegionPtr).cropRect.y = p.y - offsetY;

	    // center of box
	    float middleX = (*activeRegionPtr).cropRect.tl().x + ((*activeRegionPtr).cropRect.width)/2;
	    float middleY = (*activeRegionPtr).cropRect.tl().y + ((*activeRegionPtr).cropRect.height)/2;

	    // update the label data
	    //(*activeRegionPtr).label.bbox_x = ((*activeRegionPtr).cropRect.tl().x)/ (float) src.cols;
	    //(*activeRegionPtr).label.bbox_y = ((*activeRegionPtr).cropRect.tl().y)/ (float) src.rows;
	    (*activeRegionPtr).label.bbox_x = middleX / (float) src.cols;
	    (*activeRegionPtr).label.bbox_y = middleY / (float) src.rows;
	    (*activeRegionPtr).label.bbox_width = ((*activeRegionPtr).cropRect.width)/(float) src.cols;
	    (*activeRegionPtr).label.bbox_height = ((*activeRegionPtr).cropRect.height)/(float) src.rows;
    }
}


void addRegion(int type) {

    // create a new region
    Region region;

    // record type
    region.label.type = type;

    // center of box
    float middleX = activeCropRect.tl().x + (activeCropRect.width)/2;
    float middleY = activeCropRect.tl().y + (activeCropRect.height)/2;

    // record the label data
    //region.label.bbox_x = (activeCropRect.tl().x)/ (float) src.cols;
    //region.label.bbox_y = (activeCropRect.tl().y)/(float) src.rows;
    region.label.bbox_x = middleX / (float) src.cols;
    region.label.bbox_y = middleY / (float) src.rows;
    region.label.bbox_width = (activeCropRect.width)/(float) src.cols;
    region.label.bbox_height = (activeCropRect.height)/(float) src.rows;

    // record active rect
    region.cropRect = activeCropRect;

    // add region to the list
    regions.push_back(region);
}

void deleteRegion() {
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
	if ((*list_iter).selected) {
		// delete this region in the list
		list_iter = regions.erase(list_iter);
	}
    }
}


void onMouse( int event, int x, int y, int f, void* ){


    switch(event){

        case  CV_EVENT_LBUTTONDOWN  :		// left button down => start resize
                                        leftclicked=true;

                                        P1.x=x;
                                        P1.y=y;
                                        P2.x=x;
                                        P2.y=y;
                                        break;

        case  CV_EVENT_LBUTTONUP    :		// left button up => stop resize
                                        P2.x=x;
                                        P2.y=y;
                                        leftclicked=false;
                                        break;

        case  CV_EVENT_RBUTTONDOWN  :		// right button down => start move
                                        P2.x=x;
                                        P2.y=y;
					checkRegions(x,y);
					rightclicked=true;
                                        break;
        case  CV_EVENT_RBUTTONUP  :		// right button up => stop move
					rightclicked=false;
                                        break;
        case  CV_EVENT_MOUSEMOVE    :
                                        if(leftclicked || rightclicked){
                                        P2.x=x;
                                        P2.y=y;
                                        }

        default                     :   break;


    }


    if(leftclicked){
     if(P1.x>P2.x){ activeCropRect.x=P2.x;
                       activeCropRect.width=P1.x-P2.x; }
        else {         activeCropRect.x=P1.x;
                       activeCropRect.width=P2.x-P1.x; }

        if(P1.y>P2.y){ activeCropRect.y=P2.y;
                       activeCropRect.height=P1.y-P2.y; }
        else {         activeCropRect.y=P1.y;
                       activeCropRect.height=P2.y-P1.y; }

    }

    if (rightclicked) {
	moveRegion(P2);	
    }

    showRegions();
}

void readInClasses(const char *filename) 
{
    	ifstream myObjClassFile;

    	myObjClassFile.open(filename);
	if (myObjClassFile.is_open()) {

		while (!myObjClassFile.eof()) {
			string classStr;
			myObjClassFile >> classStr;
			classes.push_back(classStr);
		}
	}
	myObjClassFile.close();
	selectedClass = classes[0];
}

void printClasses()
{
    cout << "Classes (press number to select): " << endl;
    for (int i=0; i<classes.size(); i++)
    {
	cout << "(" << hex << i << ") :" << classes[i] << " ";
	cout << endl;
    }
    cout << endl;
}

void printLabels()
{
    cout << "Labels: " << endl;
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
	cout << (*list_iter).label << endl;
    }
}

void prepTrainingFolders() {
	struct stat st;

	// check if training folder exists. If not, create it
	if (stat("./yolo_training_data",&st) == -1)
	{
		cout << "Creating training folder..." << endl;
		mkdir("./yolo_training_data",0700);
	}
	
	// check if training images folder exists. If not, create it
	if (stat("./yolo_training_data/images",&st) == -1)
	{
		cout << "Creating training images folder..." << endl;
		mkdir("./yolo_training_data/images",0700);
	}

	// check if training labels folder exists. If not, create it
	if (stat("./yolo_training_data/labels", &st) == -1)
	{
		cout << "Creating training labels folder..." << endl;
		mkdir("./yolo_training_data/labels",0700);
	}	

	// set paths		
	trainingImagePath = "./yolo_training_data/images/" + trainingFilenamePrefix;
	trainingLabelPath = "./yolo_training_data/labels/" + trainingFilenamePrefix;
}

void exportTrainingData()
{
    //struct timeval timeStamp;
    //gettimeofday(&timeStamp,NULL);

    //char buf[32];
    //snprintf(buf,sizeof(buf),"_%d_%06d",(int)timeStamp.tv_sec,(int)timeStamp.tv_usec);

    // check to make sure the folders are still there before exporting
    prepTrainingFolders();

    char counterStr[6];
    sprintf(counterStr,"%06ld",frameCounter++);

    string imgFileName = trainingImagePath + counterStr;
    //imgFileName.append(buf);
    imgFileName.append(".jpg");

    string labelFileName = trainingLabelPath + counterStr;
    //labelFileName.append(buf);
    labelFileName.append(".txt");

    //cout << "imgFile = " << imgFileName <<  " labelFile = " << labelFileName << endl;

    // save original image
    imwrite(imgFileName.c_str(),src);

    // save corresponding label file
    ofstream outFile;
    outFile.open(labelFileName.c_str());
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
      	outFile << (*list_iter).label << endl;
    }	
    outFile.close();

}

int main(int argc, char* argv[])
{

    if (argc != 5)
    {
	cout << "Usage: " << argv[0] << " <filename | VIDEO> <starting_frame> <classes_file> <training_file_prefix>" << endl;
	return 0;
    }

    videoFilename = argv[1];
    VideoCapture vidCap;
    cout << "videoFilename = " << videoFilename << endl;
    if (videoFilename == "VIDEO")
	vidCap = VideoCapture(0);
    else
	vidCap = VideoCapture(videoFilename);

    //VideoCapture vidCap(videoFilename);
    if (!vidCap.isOpened()) 
	return -1;

    string startingFrameStr = argv[2];
    startingFrame = stoi(startingFrameStr);
 
    readInClasses(argv[3]);
    printClasses();

    trainingFilenamePrefix = argv[4];
    cout << "trainingFilenamePrefix = " << trainingFilenamePrefix << endl;
    prepTrainingFolders();

    cout<<"Left click and drag to define region"<<endl;
    cout<<"Right click and drag to select & move existing region"<<endl;
    cout<<"--> Press number to create new region"<<endl;
    cout<<"--> Press 'x' to delete currently active region"<<endl;
    cout<<"--> Press 'z' to delete all regions"<<endl;
    cout<<"--> Press 'f' to toggle frame advance"<<endl;
    cout<<"--> Press spacebar to toggle training data export"<<endl;

    namedWindow(winName,CV_WINDOW_AUTOSIZE);
    setMouseCallback(winName,onMouse,NULL );

    // zip forward to starting frame number (if using a file)
    if (videoFilename != "VIDEO") {
        cout << "seeking to frame #" << std::dec << startingFrame << endl;
 	vidCap.set(CV_CAP_PROP_POS_FRAMES,(double) startingFrame);
    }

    while(1){

	    if (frameAdvance) {
		    // load a frame from vidCap
		    vidCap >> src;
	    }

	    // display current regions
	    showRegions();

	    // if training export enable, save files
	    if (trainingDataExportEnable)
	    	exportTrainingData();
	
	    // check for keypress every 100 ms
	    char c = waitKey(100);

	    // if a real key was pressed...
	    if (c != -1) {

		    // space toggles training export - disables frame advance
		    if(c==' ')
		    {
			frameAdvance = false;
			trainingDataExportEnable = !trainingDataExportEnable;
		    }

		    // 'f' toggles frame advance - disables training
		    if(c=='f')
		    {
			trainingDataExportEnable = false;
			frameAdvance = !frameAdvance;
		    }

		    // delete selected region
		    if (c=='x')
		    {
			deleteRegion();
		    }

		    // delete all regions
		    if (c=='z')
		    {
			regions.clear();
		    }
		
		    // add selected region with a class type
		    if (c>='0' && c<='9')
		    {
			/*
			int i;
			
			// only ten classes allowed right now
			if (c >= '0' && c <= '9')
				i = (int) (c - '0');
		    	list<string>::iterator list_iter = classes.begin();
			advance(list_iter,i);

			selectedClass = *list_iter;
			addRegion(selectedClass);
			*/
			int i = c - '0';
			addRegion(i);
		    }

		    if(c==27) break;
		}

    }

    return 0;
}
