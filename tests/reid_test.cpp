#include "Reid.h"
#include <iostream>

int main()
{
    ReidModel model("../models/dep_models/osnet1.so");

    // Load an image using OpenCV.
    cv::Mat image = cv::imread("../data/example.jpg", cv::IMREAD_COLOR);
    std::cout<<"Image read"<< std::endl;
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }

    // Convert image to floating point.
    image.convertTo(image, CV_32F);

    // Resize the image to match the input size of the model.
    cv::resize(image, image, cv::Size(128, 256));

    // Run the model.
    std::vector<float> output = model.run(image);

    // Print out the output.
    for (auto i : output)
    {
        std::cout << i << ", ";
    }
    std::cout << "\n";

    return 0;
}
