
#include <iostream>
#include <dirent.h>
#include <iomanip>
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
	 string type;		// Describes the type of object: e.g. 'Car', 'Van', 'Truck',
                     		// 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     		// 'Misc' or 'DontCare'

	 float truncated;	// Float from 0 (non-truncated) to 1 (truncated), 
				// where truncated refers to the object leaving image boundaries

	 int   occluded;   	// Integer (0,1,2,3) indicating occlusion state:
                     		// 0 = fully visible, 1 = partly occluded
                     		// 2 = largely occluded, 3 = unknown
	 float obsAngle;	// Observation angle of object, ranging [-pi..pi]

	 float bbox_left;	// 2D bounding box of object in the image (0-based index):
         float bbox_top;	// contains left, top, right, bottom pixel coordinates
	 float bbox_right;
	 float bbox_bottom;             		
	
	 float dim_height;	// 3D object dimensions: height, width, length (in meters)
	 float dim_width;
	 float dim_length;
 
	 float loc_x;		// 3D object location x,y,z in camera coordinates (in meters)
	 float loc_y;
	 float loc_z;

	 float rot_y;		// Rotation ry around Y-axis in camera coordinates [-pi..pi]

	 //float score;		// Only for results: Float, indicating confidence in
         //             	// detection, needed for p/r curves, higher is better.

	 LabelEntry() {
	    type = "DontCare";
	    truncated = 0.0f;
	    occluded = 0;
	    obsAngle = 0.0f;
	    bbox_left = 0.0f;
	    bbox_top = 0.0f;
	    bbox_right = 0.0f;
	    bbox_bottom = 0.0f;
	    dim_height = 0.0f;
	    dim_width = 0.0f;
	    dim_length = 0.0f;
	    loc_x = 0.0f;
	    loc_y = 0.0f;
	    loc_z = 0.0f;
	    rot_y = 0.0f;
	}
};	
	
std::ostream& operator<<(std::ostream& out, const LabelEntry& obj) {
	return out << obj.type << " " 
		   << obj.truncated << " " 
		   << obj.occluded << " " 
		   << obj.obsAngle << " " 
		   << obj.bbox_left << " " 
		   << obj.bbox_top << " " 
	   	   << obj.bbox_right << " "
		   << obj.bbox_bottom << " " 
		   << obj.dim_height << " "
		   << obj.dim_width << " "
		   << obj.dim_length << " "
		   << obj.loc_x << " "
		   << obj.loc_y << " "
		   << obj.loc_z << " "
		   << obj.rot_y;
}

std::istream& operator >>(std::istream &is, LabelEntry &obj)
{
	is >> obj.type
	   >> obj.truncated
           >> obj.occluded 
	   >> obj.obsAngle 
	   >> obj.bbox_left 
	   >> obj.bbox_top
	   >> obj.bbox_right
	   >> obj.bbox_bottom 
	   >> obj.dim_height
	   >> obj.dim_width
	   >> obj.dim_length
	   >> obj.loc_x
	   >> obj.loc_y
	   >> obj.loc_z
	   >> obj.rot_y;

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
list<string> classes;
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
	putText(img,(*list_iter).label.type,pt,fontFace,fontScale,cvScalar(200,200,250),thickness,8, false);

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

	    // update the label data
	    (*activeRegionPtr).label.bbox_left = (*activeRegionPtr).cropRect.tl().x;
	    (*activeRegionPtr).label.bbox_top = (*activeRegionPtr).cropRect.tl().y;
	    (*activeRegionPtr).label.bbox_right = (*activeRegionPtr).cropRect.br().x;
	    (*activeRegionPtr).label.bbox_bottom = (*activeRegionPtr).cropRect.br().y;
    }
}


void addRegion(string typeName) {

    // create a new region
    Region region;

    // record type
    region.label.type = typeName;

    // record the label data
    region.label.bbox_left = activeCropRect.tl().x;
    region.label.bbox_top = activeCropRect.tl().y;
    region.label.bbox_right = activeCropRect.br().x;
    region.label.bbox_bottom = activeCropRect.br().y;

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

    // record region rect
    Point topLeft(label.bbox_left, label.bbox_top);
    Point bottomRight(label.bbox_right, label.bbox_bottom);
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

	list<string>::iterator list_iter = classes.begin(); 
	selectedClass = *list_iter;
}

void printClasses()
{
    cout << "Classes (press number to select): " << endl;
    int i = 0;
    for (list<string>::iterator list_iter = classes.begin(); list_iter != classes.end(); list_iter++)
    {
	cout << "(" << hex << i++ << ") :" << *list_iter << " ";
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
			int i;
			
			// only ten classes allowed right now
			if (c >= '0' && c <= '9')
				i = (int) (c - '0');
		    	list<string>::iterator list_iter = classes.begin();
			advance(list_iter,i);

			selectedClass = *list_iter;

			addRegion(selectedClass);
		    }
		
		    if(c==27) break;
		}

    }

    return 0;
}
