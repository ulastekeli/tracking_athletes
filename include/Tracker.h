#ifndef TRACKER_H
#define TRACKER_H

#include "Track.h"
#include "Detector.h"
#include "Reid.h"
#include <vector>
#include <opencv2/opencv.hpp>

class Tracker {
public:
    Tracker(const std::string& reid_model_path);
    
    void match(const std::vector<DetectedObject>& detections, const cv::Mat& frame);

private:
    std::vector<Track> tracks;
    int id_counter;
    int frame_no;
    ReidModel reid;
};

#endif // TRACKER_H
