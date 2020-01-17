# YoloV3 Darknet Multi-Object Detection and Classification

A first look into Real-Time Multi-Object Detection and Classification, algorithms using Python and C++, from a direct Webcam or Video input.

The YoloV3 uses a CNN Architecture called Darknet-53 an evolution from YoloV2 architecture Darknet-19, and this algorithm uses an 80 classes trained model from ImageNet, that can be found in the second reference below, this algorithm do only the inference through the model.

![Algorithm Detection](https://i.ibb.co/mtNyKQT/iss.png)


## References

This algorithm is mostly if not entirely, based articles in the following links.

- [Learn OpenCV Website](https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/)
- [Pjreddie Website](https://pjreddie.com/darknet/yolo/)
- [Embedded Vision Website](https://www.embedded-vision.com/academy/Embedded_Vision_Alliance_Meetup_March_2019_OpenCV.pdf)
- [Towards Data Science](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

## Libraries

### Python

- OpenCV2
- Numpy

### C++



## Observations

- Due to perfomance issues, the input used are 96x96.
- Backend are CPU and OpenCV based.
- Bounding bozes only to hits above 70% of confidence.

