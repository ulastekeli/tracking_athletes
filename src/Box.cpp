// Box.cpp
#include "Box.h"

Box::Box(float x_min, float y_min, float x_max, float y_max, float conf)
    : xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), confidence(conf)
{
}

void Box::print() const {
    std::cout << "Box: (" << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "), Confidence: " << confidence << std::endl;
}

void Box::checkBounds(int image_width, int image_height) {
    xmin = std::max(0.0f, xmin);
    ymin = std::max(0.0f, ymin);
    xmax = std::min(static_cast<float>(image_width), xmax);
    ymax = std::min(static_cast<float>(image_height), ymax);
}
