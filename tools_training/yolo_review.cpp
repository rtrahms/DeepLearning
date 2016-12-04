
#include <iostream>
#include <dirent.h>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
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

	 int type;
	 float bbox_x;	// 2D bounding box of object in the image (0-based index):
         float bbox_y;	// contains left, top, right, bottom pixel coordinates
	 float bbox_width;
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

std::istream& operator >>(std::istream &is, LabelEntry &obj)
{
	is >> obj.type
	   >> obj.bbox_x 
	   >> obj.bbox_y
	   >> obj.bbox_width
	   >> obj.bbox_height;

	return is;
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

// training data file vars
int imgIndex;
string trainingLabelPath;
string labelFileName;
vector<string> labelFileNames;

string trainingImagePath;
string imageFileName;
vector<string> imageFileNames;

const char* winName="Crop Image";
bool leftclicked=false;
bool rightclicked=false;
char imgName[15];

// returns a list of files from directory
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);

    // sort files list alphabetically
    std::sort( files.begin(), files.end() );

    return 0;
}

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

    Point pt2(img.cols/2 - 20,img.rows - 20);
    putText(img,imageFileNames[imgIndex],pt2,fontFace,1,cvScalar(255,255,255), 2 ,8, false);

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

void addRegion(LabelEntry label) {

    // create a new region
    Region region;

    // record label
    region.label = label;

    // convert middle of box to topLeft coords
    float topLeftX = label.bbox_x - label.bbox_width/2;
    float topLeftY = label.bbox_y - label.bbox_height/2;

    // record region rect
    //Point topLeft(label.bbox_x * (float) src.cols, label.bbox_y * (float) src.rows);
    //Point bottomRight((label.bbox_x + label.bbox_width) * (float) src.cols, 
	//	      (label.bbox_y + label.bbox_height) * (float) src.rows);
    Point topLeft(topLeftX * (float) src.cols, topLeftY * (float) src.rows);
    Point bottomRight((topLeftX + label.bbox_width) * (float) src.cols, 
		      (topLeftY + label.bbox_height) * (float) src.rows);
    region.cropRect = Rect(topLeft, bottomRight);

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

void importTrainingData(int index)
{
    // first, clear the regions list
    regions.clear();

    string tempImageFileName = trainingImagePath + "/" + imageFileNames[index];
    string tempLabelFileName = trainingLabelPath + "/" + labelFileNames[index];

    cout << "imgFileName = " << imageFileNames[index] << endl;
    cout << "labelFilename = " << labelFileNames[index] << endl;
    cout << "imgFilePath = " << tempImageFileName << endl;
    cout << "labelFilePath = " << tempLabelFileName << endl;

    // load image from file
    src = imread(tempImageFileName.c_str(), -1);

    // read in corresponding label file
    ifstream inFile;
    inFile.open(tempLabelFileName.c_str());
     
    while (!inFile.eof()) {
    	LabelEntry newLabel;
    
	inFile >> newLabel;
    	addRegion(newLabel);
	
    }

    inFile.close();
}

void deleteTrainingData(int index)
{
    string tempImageFileName = trainingImagePath + "/" + imageFileNames[index];
    string tempLabelFileName = trainingLabelPath + "/" + labelFileNames[index];

    //cout << "imgFile = " << imgFileName <<  " labelFile = " << labelFileName << endl;

    // remove image file
    remove(tempImageFileName.c_str());
    remove(tempLabelFileName.c_str());

    // clear the regions list
    regions.clear();

    // remove the names from the lists
    imageFileNames.erase(imageFileNames.begin() + index);
    labelFileNames.erase(labelFileNames.begin() + index);
}

void exportTrainingData(int index)
{
    string tempImageFileName = trainingImagePath + "/" + imageFileNames[index];
    string tempLabelFileName = trainingLabelPath + "/" + labelFileNames[index];

    // save original image - Not needed to resave image file, just labels
    //imwrite(tempImageFileName.c_str(),src);

    // overwrite corresponding label file
    //cout << "tempLabelFileName = " << tempLabelFileName << endl;
    ofstream outFile;
    outFile.open(tempLabelFileName.c_str());
    for (list<Region>::iterator list_iter = regions.begin(); list_iter != regions.end(); list_iter++)
    {
      	outFile << (*list_iter).label << endl;
    }	
    outFile.close();

}

int main(int argc, char* argv[])
{

    if (argc != 4)
    {
	cout << "Usage: " << argv[0] << " <trainingImagePath> <trainingLabelPath> <classes_file>" << endl;
	return 0;
    }

    trainingImagePath = argv[1];
    imageFileNames = vector<string>();
    if (getdir(trainingImagePath,imageFileNames) != 0)
    {
	cout << "Error reading image files from " << trainingImagePath << endl;
	return -1;
    }

    trainingLabelPath = argv[2];
    labelFileNames = vector<string>();
    if (getdir(trainingLabelPath,labelFileNames) != 0)
    {
	cout << "Error reading label files from " << trainingLabelPath << endl;
	return -1;
    }

    cout << "trainingLabelPath = " << trainingLabelPath << ": number of files = " << imageFileNames.size() << endl;
    cout << "trainingImagePath = " << trainingImagePath << ": number of files = " << labelFileNames.size() << endl;

    readInClasses(argv[3]);
    printClasses();

    cout<<"Left click to select region"<<endl;
    cout<<"Right click and drag to select & move existing region"<<endl;
    cout<<"--> Press number to create new region"<<endl;
    cout<<"--> Press 'a' or 'd' to move between training images"<<endl;
    cout<<"--> Press 's' to re-save current training image/data"<<endl;
    cout<<"--> Press 'x' to delete currently active region"<<endl;
    cout<<"--> Press 'z' to delete all regions"<<endl;
    cout<<"--> Press 'p' to purge current training image/data"<<endl;

    namedWindow(winName,CV_WINDOW_AUTOSIZE);
    setMouseCallback(winName,onMouse,NULL );

    // load first image and data - after "." (0) and ".." (1)
    imgIndex = 2;
    importTrainingData(imgIndex);

    while(1){

	    // display current regions
	    showRegions();
	
	    // check for keypress every 100 ms
	    char c = waitKey(100);

	    // if a real key was pressed...
	    if (c != -1) {

		    // 'd' advances to next training image
		    if(c=='d')
		    {
			if (imgIndex < imageFileNames.size() - 1) {
				importTrainingData(++imgIndex);
				cout << "Training set #" << std::setbase(10) << imgIndex << endl;
			}
		    }

		    // 'a' advances to prev training image
		    if(c=='a')
		    {
			if (imgIndex > 2) {
				importTrainingData(--imgIndex);
				cout << "Training set #" << std::setbase(10) << imgIndex << endl;
			}
		    }

		    // 's' re-saves current training image & data
		    if(c=='s')
		    {
			// re-save current training image & data
			cout << "re-saving training set #" << std::setbase(10) << imgIndex << "." << endl;
			exportTrainingData(imgIndex);
		    }

		    // 'p' deletes image/label files
		    if(c=='p')
		    {
			cout << "purge training set #" << std::setbase(10) << imgIndex << "." << endl;

			// delete current training image & label files
			deleteTrainingData(imgIndex);
			
			// load the previous image
			if (imgIndex > 2) {
				importTrainingData(--imgIndex);
				cout << "Training set #" << std::setbase(10) << imgIndex << endl;
			}
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
