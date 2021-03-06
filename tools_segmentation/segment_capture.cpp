#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <malloc.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;

// Byte order:  Blue - Green - Red - Alpha
unsigned long BGRA_color_map[21] = {
	0x00000000,	// background
	0x00008000,	// aeroplane
	0x00800000,	// bicycle
	0x00808000,	// bird
	0x80000000,	// boat
	0x80008000,	// bottle
	0x80800000,	// bus
	0x80808000,	// car
	0x00004000,	// cat
	0x0000c000,	// chair
	0x00804000,	// cow
	0x0080c000,	// diningtable
	0x80004000,	// dog
	0x8000c000,	// horse
	0x80804000,	// motorbike
	0x8080c000,	// person
	0x00400000,	// pottedplant
	0x00408000,	// sheep
	0x00c00000,	// sofa
	0x00c08000,	// train
	0x80400000	// tvmonitor
};

class Segmenter {
 public:
  Segmenter(const string& model_file,
             const string& trained_file,
             const string& label_file);

   cv::Mat CreateSegmentedImage(const cv::Mat& img, int N = 5);

 private:
  cv::Mat SegmentProcess(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};

Segmenter::Segmenter(const string& model_file,
                       const string& trained_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  std::cout << "Setting up network..." << std::endl;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  std::cout << "checking inputs, outputs..." << std::endl;

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly TBD outputs.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
   
  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
  {
    //std::cout << line << std::endl;
    labels_.push_back(string(line));
  }

  Blob<float>* output_layer = net_->output_blobs()[0];
  //CHECK_EQ(labels_.size(), output_layer->channels())
  //  << "Number of labels is different from the output layer dimension.";

  // report out on network stuff - Debug only
  
  std::cout << "Network name = " << net_->name() << std::endl;
  vector<string> layer_names = net_->layer_names();
  std::cout << "Layers (" << layer_names.size() << ")" << std::endl;
  for (int i=0; i<layer_names.size(); i++)
  	std::cout << layer_names[i] << std::endl;

  vector<string> blob_names = net_->blob_names();
  std::cout << "Blobs (" << blob_names.size() << ")" << std::endl;
  for (int i=0; i<blob_names.size(); i++)
  	std::cout << blob_names[i] << std::endl;

  std::cout << "output layer number = " << output_layer->num() << std::endl;
  std::cout << "output layer channels = " << output_layer->channels() << std::endl;
  std::cout << "output layer height = " << output_layer->height() << std::endl;
  std::cout << "output layer width = " << output_layer->width() << std::endl;
 
}

/* Return the indices of the top value in list. */
static int ArgMax(const float *values[], int ptr_increment, int size) {
	
	float maxVal = 0;
	int maxIndex = 0;   // by default return background

	for (int i=0; i < size; i++) {
		if (*(values[i]+ptr_increment) > maxVal) {
			maxVal = *(values[i]+ptr_increment);
			maxIndex = i;
		}
	}

	return maxIndex;
}

/* Return the segmented image. */
cv::Mat Segmenter::CreateSegmentedImage(const cv::Mat& img, int N) {
  cv::Mat output = SegmentProcess(img);

  return output;
}

cv::Mat Segmenter::SegmentProcess(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Grab a reference to the output layer as an float buffer ptr */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float *output_buffer = output_layer->cpu_data();
  //std::cout << "Output Blob shape = " << output_layer->shape_string() << std::endl;

  // dimensions - number of classes (C), image height (H), image width (W)
  const int num_classes = output_layer->channels();
  int image_size = output_layer->height() * output_layer->width();
 
  // set up the class buffer pointers into each class sub-buffer
  const float *class_buf_ptr[num_classes];
  for (int i=0; i<num_classes; i++) {
	class_buf_ptr[i] = &(output_buffer[i*image_size]);
  }
 
  /****************** creating RGBA image buffer from output layer ***************/
  //assume three channels only
  int num_channels = 3;
  int fill_buffer_size = image_size * num_channels;
  unsigned char fill_buffer[fill_buffer_size];
  memset(fill_buffer, 0, fill_buffer_size);

  // for each pixel in the image
  for (int i=0; i<image_size; i++) {
	
	// find the index of most-likely class for this pixel
	int topClassIndex = ArgMax(class_buf_ptr,i,num_classes);

	// apply the colormap for that class
	int bufIndex = i*num_channels;
	fill_buffer[bufIndex] = (BGRA_color_map[topClassIndex] >> 24) & 0xff;      // B
	fill_buffer[bufIndex+1] = (BGRA_color_map[topClassIndex] >> 16) & 0xff;    // G
	fill_buffer[bufIndex+2] = (BGRA_color_map[topClassIndex] >> 8) & 0xff;     // R
	//fill_buffer[bufIndex+3] = BGRA_color_map[topClassIndex] & 0xff;   	   // A
     }

  cv::Mat segmentedImg = cv::Mat(output_layer->height(), output_layer->width(), CV_8UC3, &fill_buffer[0], 0);
  /****************** creating RGBA image buffer from output layer ***************/

  return segmentedImg;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Segmenter::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Segmenter::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int Display_Text( cv::Mat image, std::string text, Point org )
{
  int lineType = 8;

  putText( image, text, org, CV_FONT_HERSHEY_COMPLEX_SMALL, 1,
           Scalar::all(255), 1, lineType );
  return 0;
}


int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " labels.txt [VIDEO|Filename]" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string label_file   = argv[3];
  string videoFilename = argv[4];

  cv::VideoCapture vidCap;
  std::cout << "videoFilename = " << videoFilename << std::endl;
  if (videoFilename == "VIDEO")
	vidCap = VideoCapture(0);
  else
	vidCap = VideoCapture(videoFilename);

    //VideoCapture vidCap(videoFilename);
    if (!vidCap.isOpened()) 
	return -1;

  // set up the classifier network
  Segmenter segmenter(model_file, trained_file, label_file);
	
  for (;;) 
  {
  	  // grab an image to process
  	  cv::Mat inputImg;
  	  vidCap >> inputImg;

	  // get segmented image
	  cv::Size size = inputImg.size();
	  cv::Mat segmentedImg = segmenter.CreateSegmentedImage(inputImg, 21);
	  cv::resize(segmentedImg,segmentedImg,size);

	  // combine input and segmented images
	  cv::Mat combinedImg = cv::Mat(size.height, size.width, CV_8UC4);
	  cv::addWeighted(inputImg, 0.5f, segmentedImg, 0.5f, 0.0f, combinedImg);

	  //imshow("Input Image", inputImg);
	  //imshow("Segmenter output", segmentedImg);
	  imshow("Combined output", combinedImg);

	  // exit on ESC key
	  if (cv::waitKey(1) == 27)
		break;
  }

  return 0;	
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
