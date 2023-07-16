#ifndef BOX_H
#define BOX_H

class Box {
public:
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;

    Box(float x_min, float y_min, float x_max, float y_max, float conf);
};

#endif // BOX_H