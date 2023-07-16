#include "Detector.h"
#include "Box.h"
#include <iostream>

int main()
{
    std::string modelWeights = "../models/dev_models/pd.weights";
    std::string modelConfiguration = "../models/dev_models/pd.cfg";
    std::string classNamesFile = "../models/dev_models/pd.names";
    
    ObjectDetector detector(modelWeights, modelConfiguration, classNamesFile);
    
    cv::Mat image = cv::imread("../data/football.jpg");
    
    if (image.empty())
    {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }
    
    std::vector<Box> detectedBoxes = detector.detectObjects(image);
    std::cout<<"no of detected boxes " << detectedBoxes.size() << std::endl;

    for (const auto& box : detectedBoxes)
    {
        cv::rectangle(image, cv::Rect(cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax)), cv::Scalar(0, 255, 0));
        // std::cout << "Box: [" << box.xmin << ", " << box.ymin << ", " << box.xmax << ", " << box.ymax << ", " << box.confidence << "]\n";
    }
    cv::imwrite("out.jpg", image);
    
    return 0;
}
