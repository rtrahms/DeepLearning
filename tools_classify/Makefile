
GCC = /usr/bin/g++
RM = rm

CFLAGS = -I/usr/include -I/usr/local/include -I$(CAFFE_HOME)/include -I$(CAFFE_HOME)/protobuf/include -I$(CUDA_HOME)/include -std=c++11 -DUSE_CUDNN -DUSE_OPENCV -DWITH_PYTHON_LAYER

LDFLAGS = -L/usr/lib -L$(CAFFE_HOME)/build/lib -L/usr/local/lib -L/usr/local/cuda/lib64 -lcaffe-nv -lglog -lgflags -lprotobuf -lcudnn -lcudart -lcublas -lcurand -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_python -lpython2.7 -lcblas -latlas

OPENCV_CFLAGS = `pkg-config opencv --cflags --libs`

all: classify_capture.bin classification.bin

clean:
	$(RM) -f *.o *.bin

classification.bin: classification.cpp
	$(GCC) -o classification.bin classification.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

classify_capture.bin: classify_capture.cpp
	$(GCC) -o classify_capture.bin classify_capture.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

