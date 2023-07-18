#include "Track.h"

const int Track::N = 15;  // Initialize the static variable

void Track::update(std::vector<float> feature, int frame_no, Box box)
{
    features.push_back(feature);
    while (features.size() > N) {
        features.erase(features.begin());
    }

    frame_history.push_back(frame_no);
    while (frame_history.size() > N) {
        frame_history.erase(frame_history.begin());
    }

    box_history.push_back(box);
    while (box_history.size() > N) {
        box_history.erase(box_history.begin());
    }
}
