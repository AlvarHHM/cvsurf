#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <numeric>
#include <array>
#include "math.h"
#include "ObstacleAvoid.h"

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
//    std::array<float,7> arr ={0.2,0.1,1,NAN,2,3,NAN};
//
//        cout << *min_element(arr.begin(), arr.end());

    VideoCapture cap(0); //capture the video from web cam
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.0);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360.0);


    ObstacleAvoid avoid;
    Mat lastFrame;
    cap.read(lastFrame);
    avoid.init(lastFrame, cap.get(CV_CAP_PROP_POS_MSEC));
    while(true){
        Mat frame;
        cap.read(frame);
        avoid.processFrame(frame, cap.get(CV_CAP_PROP_POS_MSEC));
        if (waitKey(30) == 27) {
            break;
        }
//        while(waitKey(30) != 32);
    }


    return 0;




//    namedWindow("Original", WINDOW_AUTOSIZE);
//    Mat image;
//    SURF surf(2000);
//    vector<KeyPoint> keyPoint;
//    Mat mask;
//    while (true) {
//        cap.read(image);
//        cvtColor(image, image, CV_RGB2GRAY);
//        surf(image, mask, keyPoint);
//        drawKeypoints(image, keyPoint, image, Scalar(255, 0, 0), 4);
//        cout << keyPoint.size() << endl;
//        imshow("Original", image);
//        if (waitKey(30) == 27) {
//            break;
//        }
//    }


}
