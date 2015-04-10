#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <numeric>
#include "math.h"

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
    VideoCapture cap(0); //capture the video from web cam
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.0);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360.0);

    BFMatcher bfMatcher;
    SURF surf_ui(2000, 4, 2, true, true);

    Mat lastFrame;
    double t_last = cap.get(CV_CAP_PROP_POS_MSEC);
    cap.read(lastFrame);
    Mat roi = Mat::zeros(lastFrame.rows, lastFrame.cols, CV_8UC1);
    int scrapY = lastFrame.rows / 4;
    int scrapX = lastFrame.cols / 4;
    roi(Rect(scrapX, scrapY, roi.cols - 2 * scrapX, roi.rows - 2 * scrapY)) = true;

    vector<KeyPoint> queryKP;
    Mat qdesc;
    surf_ui.operator()(lastFrame, roi, queryKP, qdesc);
    for(KeyPoint kp : queryKP){
        kp.class_id = uniqueId();
    }

    map<int, vector<KeyPoint> > kpHist;
    while(true){
        double t_curr = cap.get(CV_CAP_PROP_POS_MSEC);
        Mat currFrame;
        cap.read(currFrame);
        Mat dispim;
        cvtColor(currFrame, dispim, CV_RGB2GRAY);

        vector<KeyPoint> trainKP;
        Mat tdesc;
        surf_ui.operator()(currFrame, roi, trainKP, tdesc);
        cout << queryKP.size() << endl;
        cout << trainKP.size();

        vector<vector<DMatch> > pre_matches;
//        bfMatcher.knnMatch(qdesc, tdesc, pre_matches, 2, roi, true);
        bfMatcher.knnMatch(qdesc, tdesc, pre_matches, 2);

        vector<double> matchdist;
        vector<DMatch> matches;

        for (vector<DMatch> m : pre_matches){
            if ((m.size() ==2 and m[0].distance >= 0.8 * m[1].distance) or m[0].distance >= 0.25){
                continue;
            }
            matches.push_back(m[0]);
            KeyPoint qkp = queryKP[m[0].queryIdx];
            KeyPoint tkp = trainKP[m[0].trainIdx];
            tkp.class_id =qkp.class_id;
            matchdist.push_back(diffKP_L2(qkp, tkp));
        }

        if (matchdist.size() != 0){
            vector<double> tmp_matchdist(matchdist.begin(), matchdist.end() - matchdist.size() / 4);
            double mean = std::accumulate(tmp_matchdist.begin(), tmp_matchdist.end(), 0.0) / tmp_matchdist.size();
            double sq_sum = std::inner_product(tmp_matchdist.begin(), tmp_matchdist.end(), tmp_matchdist.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / tmp_matchdist.size() - mean * mean);
            double threshdist = mean + 2 * stdev;
            for(int i = 0; i < matches.size(); i++){
                if (matchdist[i] < threshdist){
                    matches.erase(matches.begin() + i);
                }
            }
        }

        for(int i = 0; i < matches.size(); i++){
            if (trainKP[matches[i].trainIdx].size > queryKP[matches[i].queryIdx].size){
                matches.erase(matches.begin() + i);
            }
        }



        Mat outImg;
        drawMatches(lastFrame, queryKP, currFrame, trainKP, matches, outImg);
        imshow("Original", outImg);

        lastFrame = currFrame;
        queryKP = trainKP;
        qdesc = tdesc;
        t_last = t_curr;






        if (waitKey(30) == 27) {
            break;
        }
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
