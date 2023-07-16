#ifndef TRACK_H
#define TRACK_H

#include "Box.h"
#include <vector>

class Track {
public:
    int id;
    std::vector<Box> box_history;
    std::vector<int> frame_history;
    std::vector<std::vector<float>> features;

    void update(std::vector<float> feature, int frame_no, Box box);
};

#endif // TRACK_H