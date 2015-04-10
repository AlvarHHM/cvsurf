#include "KeyPointHistory.h"

void KeyPointHistory::update(KeyPoint kp, Mat desc, double t0, double t1, double scale) {
    if (this->timehist_t1.size() > 0 and t0 == this->timehist_t1.back()){
        this->consecutive += 1;
    }else{
        this->consecutive = 1;
    }
    this->age -= 1;
    this->lastFrameIdx = 0;
    this->detects += 1;
    this->scalehist.push_back(scale);
    this->timehist_t0.push_back(t0);
    this->timehist_t1.push_back(t1);
    this->keypoint = kp;
    desc.copyTo(this->descriptor);

}

void KeyPointHistory::downdate() {
    this->age += 1;
    this->lastFrameIdx += 1;

}
