#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "Box.h"
#include <vector>

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelWeights, const std::string& modelConfiguration,
                   const std::string& classNamesFile);
    
    void loadModel();
    std::vector<Box> detectObjects(cv::Mat frame);

private:
    std::vector<std::string> getOutputsNames();
    void postProcessing(cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Box>& detectedObjects);

    std::string modelWeights;
    std::string modelConfiguration;
    std::string classNamesFile;

    std::vector<std::string> classNames;
    cv::dnn::Net net;

    float confThreshold;
    float nmsThreshold;
};
