#include "Reid.h"
#include <iostream>



int main()
{
    ReidModel model("../models/dep_models/osnet1.so");

    cv::Mat image = cv::imread("../output/cropped/149_0.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }
    image.convertTo(image, CV_32F);
    std::vector<float> output = model.run(image);

    cv::Mat image1 = cv::imread("../output/cropped/150_0.png", cv::IMREAD_COLOR);
    image1.convertTo(image1, CV_32F);
    std::vector<float> output1 = model.run(image1);

    float dist = cv::norm(output1, output);
    std::cout<<dist<<std::endl;

    // // Print out the output.
    // for (auto i : output)
    // {
    //     std::cout << i << ", ";
    // }
    // std::cout << "\n";

    return 0;
}
