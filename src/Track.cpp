#include "Track.h"

void Track::update(std::vector<float> feature, int frame_no, Box box)
{
    features.push_back(feature);
    frame_history.push_back(frame_no);
    box_history.push_back(box);
}
