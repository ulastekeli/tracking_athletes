#include "Box.h"

Box::Box(float x_min, float y_min, float x_max, float y_max, float conf)
    : xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), confidence(conf)
{
}
