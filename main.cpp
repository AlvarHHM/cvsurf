#include <iostream>
#include <ctime>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <numeric>
#include <array>
#include "math.h"
#include "ObstacleDetect.h"

using namespace cv;
using namespace std;


int uniqueId(void) {
    static volatile int i = 0;
    return __sync_add_and_fetch(&i, 1);
}

double diffKP_L2(KeyPoint kp0, KeyPoint kp1){
    return sqrt(pow((kp0.pt.x - kp1.pt.x), 2) + pow((kp0.pt.y - kp1.pt.y), 2));
}



int main(int argc, char **argv) {
    cv::initModule_features2d();
    VideoCapture cap(0); //capture the video from web cam
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.0);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360.0);
    namedWindow("Original", WINDOW_NORMAL);

    ObstacleDetect avoid;
    Mat lastFrame;
    cap.read(lastFrame);
    cvtColor(lastFrame, lastFrame, CV_RGB2GRAY);
    avoid.init(lastFrame);

    while(true){
        Mat frame;
        cap.read(frame);
        cvtColor(frame, frame, CV_RGB2GRAY);
        avoid.processFrame(frame);
        cv::imshow("Original", frame);
        if (waitKey(1) == 27) {
            break;
        }
//        while(waitKey(30) != 32);
    }


    return 0;



}
