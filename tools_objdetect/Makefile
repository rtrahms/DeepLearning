
GCC = /usr/bin/g++
RM = rm

CFLAGS = -I/usr/include -I/usr/local/include -I$(CAFFE_HOME)/include -I$(CAFFE_HOME)/protobuf/include -I$(CUDA_HOME)/include -std=c++11 -DUSE_CUDNN -DUSE_OPENCV -DWITH_PYTHON_LAYER

LDFLAGS = -L/usr/lib -L$(CAFFE_HOME)/build/lib -L/usr/local/lib -L/usr/local/cuda/lib64 -lcaffe-nv -lglog -lgflags -lprotobuf -lcudnn -lcudart -lcublas -lcurand -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_python -lpython2.7 -lcblas -latlas

OPENCV_CFLAGS = `pkg-config opencv --cflags --libs`

all: segment_capture.bin segment_file.bin detectnet_capture.bin detectnet_file.bin

clean:
	$(RM) -f *.o *.bin

segment_capture.bin: segment_capture.cpp
	$(GCC) -o segment_capture.bin segment_capture.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

detectnet_capture.bin: detectnet_capture.cpp
	$(GCC) -o detectnet_capture.bin detectnet_capture.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

segment_file.bin: segment_file.cpp
	$(GCC) -o segment_file.bin segment_file.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

detectnet_file.bin: detectnet_file.cpp
	$(GCC) -o detectnet_file.bin detectnet_file.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

