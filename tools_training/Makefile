
GCC = /usr/bin/g++
RM = rm

CFLAGS = -I/usr/include -I/usr/local/include -std=c++11
LDFLAGS = -L/usr/lib -L/usr/local/lib -lboost_system -lboost_filesystem

OPENCV_CFLAGS = `pkg-config opencv --cflags --libs`

all: live_trainer.bin video_trainer.bin image_review.bin yolo_trainer.bin yolo_review.bin

clean:
	$(RM) -f *.o *.bin

live_trainer.bin: live_trainer.cpp
	$(GCC) -o live_trainer.bin live_trainer.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

video_trainer.bin: video_trainer.cpp
	$(GCC) -o video_trainer.bin video_trainer.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

image_review.bin: image_review.cpp
	$(GCC) -o image_review.bin image_review.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

yolo_trainer.bin: yolo_trainer.cpp
	$(GCC) -o yolo_trainer.bin yolo_trainer.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

yolo_review.bin: yolo_review.cpp
	$(GCC) -o yolo_review.bin yolo_review.cpp $(OPENCV_CFLAGS) $(CFLAGS) $(LDFLAGS) 

