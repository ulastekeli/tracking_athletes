#include "TvmDetector.h"
#include "Box.h"
#include <iostream>

int main()
{
    std::string modelPath = "../models/dep_models/tiny3_tennis_detector.so";
    
    TvmDetector detector(modelPath);
    
    cv::Mat image = cv::imread("../data/example.jpg");
    
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }
    
    std::vector<Box> detectedBoxes = detector.run(image);
    std::cout<<"no of detected boxes " << detectedBoxes.size() << std::endl;

    for (const auto& box : detectedBoxes)
    {
        cv::rectangle(image, cv::Rect(cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax)), cv::Scalar(0, 255, 0));
        std::cout << "Box: [" << box.xmin << ", " << box.ymin << ", " << box.xmax << ", " << box.ymax << ", " << box.confidence << "]\n";
    }
    cv::imwrite("out.jpg", image);
    
    return 0;
}
