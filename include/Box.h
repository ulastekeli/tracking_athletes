// Box.h
#ifndef BOX_H
#define BOX_H

#include <iostream>

class Box {
public:
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;

    Box(float x_min, float y_min, float x_max, float y_max, float conf);
    void print() const;
    void checkBounds(int image_width, int image_height);
};

#endif // BOX_H
