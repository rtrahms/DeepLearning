#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
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

class DetectNet {
 public:
  DetectNet(const string& model_file,
             const string& trained_file);

  std::vector<Rect> CreateDetections(const cv::Mat& img, int N = 5);

 private:
  std::vector<float> DetectionProcess(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};

DetectNet::DetectNet(const string& model_file,
                       const string& trained_file) {
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
   
  Blob<float>* output_layer = net_->output_blobs()[0];

  // report out on network stuff - Debug only
  /*
  std::cout << "Network name = " << net_->name() << std::endl;
  vector<string> layer_names = net_->layer_names();
  std::cout << "Layers (" << layer_names.size() << ")" << std::endl;
  for (int i=0; i<layer_names.size(); i++)
  	std::cout << layer_names[i] << std::endl;

  vector<string> blob_names = net_->blob_names();
  std::cout << "Blobs (" << blob_names.size() << ")" << std::endl;
  for (int i=0; i<blob_names.size(); i++)
  	std::cout << blob_names[i] << std::endl;
  */

}

/* Return the top N detections, resized to current image size. */
std::vector<Rect> DetectNet::CreateDetections(const cv::Mat& img, int N) {
  std::vector<float> output = DetectionProcess(img);

  std::cout << "output vector size = " << output.size() << std::endl;
  for (int i=0; i<output.size(); i++)
 	std::cout << "output[" << i << "] = " << output[i] << std::endl;

  // to calibrate on current image dims
  float xResizeFactor = (float) img.cols/input_geometry_.width;
  float yResizeFactor = (float) img.rows/input_geometry_.height;
  std::cout << "Image resize factors:  x = " << xResizeFactor << ", y = " << yResizeFactor << std::endl;

  std::vector<Rect> detections;

  int stride = 5;
  int ctr = 0;
  float conf = output[ctr+4];
  while (conf > 0) {
        float xl = output[ctr] * xResizeFactor;
	float yt = output[ctr+1] * yResizeFactor;
	float xr = output[ctr+2] * xResizeFactor;
	float yb = output[ctr+3] * yResizeFactor;
	float width = xr - xl;
	float height = yb - yt;

	cv::Rect rect(xl, yt, width, height);
	detections.push_back(rect);

	ctr += stride;
	conf = output[ctr+4];
  }

  return detections;
}

std::vector<float> DetectNet::DetectionProcess(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();

  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void DetectNet::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void DetectNet::Preprocess(const cv::Mat& img,
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
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];

  // set up the detection network
  DetectNet detectNet(model_file, trained_file);

  string file = argv[3];

  cv::Mat inputImg = cv::imread(file, -1);
  CHECK(!inputImg.empty()) << "Unable to decode image " << file;

  //imshow("Input Image", inputImg);

  std::cout << "Detection processing... " << std::endl;

  // get detections
  std::vector<Rect> detections = detectNet.CreateDetections(inputImg, 21);

  // create and show new image with detections
  cv::Mat detectedImg = inputImg;
  std::cout << "num detections = " << detections.size() << std::endl;
  for (int i=0; i<detections.size(); i++)
  {
   	cv::rectangle(detectedImg, detections[i], Scalar(0,0,255), 2, 8, 0 );
  }
  imshow("Obj Detector output", detectedImg);

  // run forever to continue showing window
  for (;;) 
  {
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
