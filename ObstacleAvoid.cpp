#include "ObstacleAvoid.h"

//const std::array<double,21> ObstacleAvoid::SCALE_RANGE = {1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175,
//                                                         1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375,
//                                                         1.4, 1.425, 1.45, 1.475, 1.5};
const std::array<double,ObstacleAvoid::SEARCH_RES+1> ObstacleAvoid::SCALE_RANGE = []{
    std::array<double, ObstacleAvoid::SEARCH_RES+1> arr;
    for(int i = 0; i < arr.size(); ++i){
        arr[i] = 1 + (i / (2.0 * ObstacleAvoid::SEARCH_RES));
    }
    return arr;
}();
ObstacleAvoid::~ObstacleAvoid() {
    delete this->bfMatcher;
    delete this->surf_ui;
}

ObstacleAvoid::ObstacleAvoid() {
    this->bfMatcher = new BFMatcher();
    this->surf_ui = new SURF(2000, 4, 2, true, true);

}

void ObstacleAvoid::processFrame(Mat& frame, double time) {
//    if(this->queryKP == nullptr){
//        init(frame, time);
//        return;
//    }

    double t_curr = time;
    Mat& currFrame = frame;


    vector<KeyPoint>& queryKP = this->queryKP;
    vector<KeyPoint> trainKP;
    Mat& qdesc = this->qdesc;
    Mat tdesc;
    auto& kphist = this->kphist;
    Mat& roi = this->roi;
    for(KeyPoint& kp : this->queryKP){
        if (kp.class_id == 1 or kp.class_id == -1) {
            kp.class_id = this->uniqueId();
        }
    }
    
    this->surf_ui->operator()(currFrame, roi, trainKP, tdesc);

    vector<vector<DMatch> > pre_matches;
    if (!(qdesc.total() == 0 or tdesc.total() == 0)) {
        this->bfMatcher->knnMatch(qdesc, tdesc, pre_matches, 2);
    }
    

    vector<double> matchdist;
    vector<DMatch> matches;

    for (vector<DMatch>& m : pre_matches){
        if ((m.size() ==2 and m[0].distance >= 0.8 * m[1].distance) or m[0].distance >= 0.25){
            continue;
        }
        matches.push_back(m[0]);
        KeyPoint& qkp = queryKP[m[0].queryIdx];
        KeyPoint& tkp = trainKP[m[0].trainIdx];
        tkp.class_id = qkp.class_id;
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

    remove_if(matches.begin(), matches.end(), [&trainKP, &queryKP](DMatch m){
        return !(trainKP[m.trainIdx].size > queryKP[m.queryIdx].size);
    });
    
    
    vector<DMatch> scaledMatches;
    vector<double> kpscales;
    this->estimateKeypointExpansion(currFrame, lastFrame, matches, queryKP,trainKP,
                              kphist, kpscales, scaledMatches);
    
    
    for(int i = 0; i < scaledMatches.size(); ++i){
        DMatch m = scaledMatches[i];
        double scale = kpscales[i];
        int clsid = trainKP[m.trainIdx].class_id;
        double t_A;
        if (kphist.count(clsid) == 0) {
            kphist[clsid] = KeyPointHistory();
            t_A = t_last;
        }else {
            t_A = kphist[clsid].timehist_t1.back();
        }
        kphist[clsid].update(trainKP[m.trainIdx], tdesc.row(m.trainIdx), currFrame, t_A, t_curr, scale);
    }

    unordered_set<int> detected;
    for (KeyPoint& kp : trainKP) {
        detected.insert(kp.class_id);
    }
    
    vector<int> deathByAge;
    for (auto& it : kphist) {
        it.second.downdate();
        if (it.second.age > 10) {
            deathByAge.push_back(it.first);
            continue;
        }
        if (detected.count(it.first) == 0 && it.second.age > 0) {
            trainKP.push_back(it.second.keypoint);
            tdesc.push_back(it.second.descriptor);
        }
    }
    
    for_each(deathByAge.begin(), deathByAge.end(), [&kphist](int id){kphist.erase(id);});
    
    if (scaledMatches.size() > 1) {
        cout << scaledMatches.size() << endl;
    }


    Mat outImg;
    vector<KeyPoint> scaledKeyPoint;
    for(DMatch& m : scaledMatches){
        scaledKeyPoint.push_back(trainKP[m.trainIdx]);
    }
    drawKeypoints(currFrame, scaledKeyPoint, outImg, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    drawMatches(this->lastFrame, queryKP, currFrame, trainKP, scaledMatches, outImg);
    imshow("Original", outImg);

    this->lastFrame = currFrame;
    this->t_last = t_curr;
    this->queryKP = trainKP;
    this->qdesc = tdesc;

}

//void ObstacleAvoid::estimateKeypointExpansion(Mat const &currFrame, Mat const &lastFrame, vector<DMatch> const &matches,
//                                              vector<KeyPoint> const &queryKPs, vector<KeyPoint> const &trainKPs,
//                                              unordered_map<int, KeyPointHistory>  &kphist,
//                                              vector<double>& scale_argmin, vector<DMatch>& expandingMatches ) {
//    const Mat& trainImg = currFrame;
//    const Mat& prevImg = lastFrame;
//    
//    for(const DMatch& m : matches){
//        const KeyPoint& qkp = queryKPs[m.queryIdx];
//        const KeyPoint& tkp = trainKPs[m.trainIdx];
//    
//        const Mat& queryImg = (kphist.count(qkp.class_id) == 1)?kphist[qkp.class_id].frame:prevImg;
//        
//        int x_qkp = qkp.pt.x;
//        int y_qkp = qkp.pt.y;
//        int r_qkp = static_cast<int>(qkp.size * 1.2 / 9 * 20 / 2);
//        int x0, y0, x1, y1;
//        const Size& qsize = queryImg.size();
//        trunc_coords(qsize, (x_qkp - r_qkp), (y_qkp - r_qkp), x0, y0);
//        trunc_coords(qsize, (x_qkp + r_qkp), (y_qkp + r_qkp), x1, y1);
//        Mat querypatch;
//        queryImg(Range(min(y0, y1), max(y0, y1)), Range(min(x0, x1), max(x0, x1))).copyTo(querypatch);
//        if (querypatch.total() == 0) continue;
//        normalizeMAtrix(querypatch, querypatch);
//        
//        
//        std::array<double, SCALE_RANGE.size()> tm_scale;
//        std::fill(tm_scale.begin(), tm_scale.end(), NAN);
//        for(int i = 0; i < SCALE_RANGE.size(); ++i){
//            double scale = SCALE_RANGE[i];
//            
//            int x_tkp = tkp.pt.x;
//            int y_tkp = tkp.pt.y;
//            int r_tkp = static_cast<int>(tkp.size * 1.2 / 9 * 20 * scale / 2);
//            int x0, y0, x1, y1;
//            Size tsize = trainImg.size();
//            trunc_coords(tsize, (x_tkp - r_tkp), (y_tkp - r_tkp), x0, y0);
//            trunc_coords(tsize, (x_tkp + r_tkp), (y_tkp + r_tkp), x1, y1);
//            Mat scaledtrain;
//            trainImg(Range(min(y0, y1), max(y0, y1)), Range(min(x0, x1), max(x0, x1))).copyTo(scaledtrain);
//            if (scaledtrain.total() == 0) continue;
//            normalizeMAtrix(scaledtrain, scaledtrain);
//            
//            Mat scaledquery;
//            resize(querypatch, scaledquery, scaledtrain.size(), scale, scale);
//            
//            Mat tmp;
//            pow((scaledquery - scaledtrain), 2, tmp);
//            tmp = tmp / pow(scale, 2);
//            tm_scale[i] = sum(tmp).val[0];
//        }
//        
//        if (all_of(tm_scale.begin(), tm_scale.end(), [](double i){return isnan(i);})) continue;
//        int minIndex = 0;
//        int minValue = tm_scale[0];
//        for(int i = 0; i < tm_scale.size(); i++){
//            if(tm_scale[i] < minValue){
//                minIndex = i;
//                minValue = tm_scale[i];
//            }
//        }
//        if ((SCALE_RANGE[minIndex] > MINSIZE) and (minValue < 0.8 * tm_scale[0])){
//            scale_argmin.push_back(SCALE_RANGE[minIndex]);
//            expandingMatches.push_back(m);
//        }
//
//    }
//
//}


void ObstacleAvoid::estimateKeypointExpansion(Mat const &currFrame, Mat const &lastFrame, vector<DMatch> const &matches,
                                              vector<KeyPoint> const &queryKPs, vector<KeyPoint> const &trainKPs,
                                              unordered_map<int, KeyPointHistory>  &kphist,
                                              vector<double>& scale_argmin, vector<DMatch>& expandingMatches ) {

    const Mat& trainImg = currFrame;
    const Mat& prevImg = lastFrame;

    for(const DMatch& m : matches){
        const KeyPoint& qkp = queryKPs[m.queryIdx];
        const KeyPoint& tkp = trainKPs[m.trainIdx];

        const Mat& queryImg = (kphist.count(qkp.class_id) == 1)?kphist[qkp.class_id].frame:prevImg;

        int x_qkp = static_cast<int>(qkp.pt.x);
        int y_qkp = static_cast<int>(qkp.pt.y);
        int qkpr = static_cast<int>(qkp.size * KEYPOINT_SCALE / 2);
        int x0, y0, x1, y1;
        trunc_coords(queryImg.size(), x_qkp - qkpr, y_qkp - qkpr, x0, y0);
        trunc_coords(queryImg.size(), x_qkp + qkpr, y_qkp + qkpr, x1, y1);
        Mat querypatch;
        queryImg(cv::Range(min(y0, y1), max(y0, y1)), cv::Range(min(x0, x1), max(x0, x1))).copyTo(querypatch);
        if (querypatch.total() == 0) continue;
        normalizeMAtrix(querypatch, querypatch);
        
        int x_tkp = static_cast<int>(tkp.pt.x);
        int y_tkp = static_cast<int>(tkp.pt.y);
        int tkpr = static_cast<int>(tkp.size * KEYPOINT_SCALE * SCALE_RANGE.back() / 2);
        trunc_coords(trainImg.size(), x_tkp - tkpr, y_tkp - tkpr, x0, y0);
        trunc_coords(trainImg.size(), x_tkp + tkpr, y_tkp + tkpr, x1, y1);
        Mat trainpatch;
        trainImg(cv::Range(min(y0, y1), max(y0, y1)), cv::Range(min(x0, x1), max(x0, x1))).copyTo(trainpatch);
        normalizeMAtrix(trainpatch, trainpatch);
        
        
        std::array<double, SCALE_RANGE.size()> res;
        std::fill(res.begin(), res.end(), NAN);
        x_tkp = x_tkp-x0;
        y_tkp = y_tkp-y0;

        for (int i = 0; i < SCALE_RANGE.size(); ++i){
            double scale = SCALE_RANGE[i];
            int r = static_cast<int>(qkp.size * KEYPOINT_SCALE * scale / 2);
            trunc_coords(trainpatch.size(), x_tkp - r, y_tkp - r, x0, y0);
            trunc_coords(trainpatch.size(), x_tkp + r, y_tkp + r, x1, y1);
            Mat scaledtrain = trainpatch(cv::Range(min(y0, y1), max(y0, y1)), cv::Range(min(x0, x1), max(x0, x1)));
            if (scaledtrain.total() == 0) continue;

            Mat scaledquery;
            resize(querypatch, scaledquery, scaledtrain.size(), scale, scale, cv::INTER_LINEAR);

            Mat tmp;
            pow((scaledquery - scaledtrain), 2, tmp);
            res[i] = static_cast<double>(cv::sum(tmp)[0] / pow(scale, 2));
        }


        if (all_of(res.begin(), res.end(), [](double i){return std::isnan(i);})) continue;
        int minIndex = 0;
        int minValue = res[0];
        for(int i = 0; i < res.size(); i++){
            if(res[i] < minValue){
                minIndex = i;
                minValue = res[i];
            }
        }
        if ((SCALE_RANGE[minIndex] > MINSIZE) and (minValue < 0.8 * res[0])){
            scale_argmin.push_back(SCALE_RANGE[minIndex]);
            expandingMatches.push_back(m);
        }
    }
}

void ObstacleAvoid::trunc_coords(const Size& dims, const int& in_x, const int& in_y, int& out_x, int& out_y) {
    out_x = (in_x >= 0 and in_x <= dims.width)? in_x: (in_x < 0)? 0: dims.width;
    out_y = (in_y >= 0 and in_y <= dims.height)? in_y: (in_y < 0)? 0: dims.height;
}


void ObstacleAvoid::init(Mat& frame, double time) {
    this->lastFrame = frame;
    this->t_last = time;
    Mat roi = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    int scrapY = frame.rows / 4;
    int scrapX = frame.cols / 4;
    roi(Rect(scrapX, scrapY, roi.cols - 2 * scrapX, roi.rows - 2 * scrapY)) = true;
    this->roi = roi;
    this->qdesc = Mat();

    this->surf_ui->operator()(frame, roi, this->queryKP, this->qdesc);
    for(auto& kp : this->queryKP){
        kp.class_id = this->uniqueId();
    }
}


int ObstacleAvoid::uniqueId(void) {
    static volatile int i = 2;
    return __sync_add_and_fetch(&i, 1);
}

void ObstacleAvoid::normalizeMAtrix(const Mat& inImg, Mat& outImg){
    Mat tmp;
    inImg.convertTo(tmp, CV_32F);
    Scalar mean_scala;
    Scalar stddev_scala;
    meanStdDev(tmp, mean_scala, stddev_scala);
    double mean = static_cast<double>(mean_scala.val[0]);
    double std = static_cast<double>(stddev_scala.val[0]);

    outImg = (tmp - mean) / std;
}

//void ObstacleAvoid::normalizeMAtrix(const Mat& inImg, Mat& outImg){
//    Mat tmp;
//    inImg.convertTo(tmp, CV_32F);
//    double min, max;
//    minMaxLoc(tmp, &min, &max);
//    outImg = (tmp - min) / (max - min);
//}



double ObstacleAvoid::diffKP_L2(KeyPoint kp0, KeyPoint kp1){
    return sqrt(pow((kp0.pt.x - kp1.pt.x), 2) + pow((kp0.pt.y - kp1.pt.y), 2));
}


