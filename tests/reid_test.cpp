#include "Reid.h"
#include <iostream>

float meanAbsDiff(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors are not the same size." << std::endl;
        return -1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < vec1.size(); i++) {
        sum += std::abs(vec1[i] - vec2[i]);
    }

    return sum / vec1.size();
}

int main()
{
    ReidModel model("../models/dep_models/osnet1.so");

    cv::Mat image = cv::imread("../data/reid/149_0.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }
    image.convertTo(image, CV_32F);
    std::vector<float> output = model.run(image);

    cv::Mat image1 = cv::imread("../data/reid/150_2.png", cv::IMREAD_COLOR);
    image1.convertTo(image1, CV_32F);
    std::vector<float> output1 = model.run(image1);



    cv::Mat image_dif = cv::imread("../data/reid/150_0.png", cv::IMREAD_COLOR);
    image_dif.convertTo(image_dif, CV_32F);
    std::vector<float> output_dif = model.run(image_dif);


    float dist = meanAbsDiff(output1, output);
    float dist2 = cv::norm(output1, output);
    
    float dist_dif = meanAbsDiff(output, output_dif);
    float dist_dif2 = cv::norm(output, output_dif);
    std::cout<<dist<<std::endl;
    std::cout<<dist2<<std::endl;
    std::cout<<dist_dif<<std::endl;
    std::cout<<dist_dif2<<std::endl;

    // // Print out the output.
    // for (auto i : output)
    // {
    //     std::cout << i << ", ";
    // }
    // std::cout << "\n";

    return 0;
}
