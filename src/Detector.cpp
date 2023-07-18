#include "Detector.h"
#include <fstream>

ObjectDetector::ObjectDetector(const std::string& modelWeights, const std::string& modelConfiguration)
    : modelWeights(modelWeights), modelConfiguration(modelConfiguration), 
      confThreshold(0.4), nmsThreshold(0.2)
{
    loadModel();
}

void ObjectDetector::loadModel()
{

    // Load the network
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Box> ObjectDetector::detectObjects(cv::Mat frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(960, 544), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames());

    std::vector<Box> detectedBoxes;
    postProcessing(frame, outs, detectedBoxes);

    return detectedBoxes;
}

std::vector<std::string> ObjectDetector::getOutputsNames()
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void ObjectDetector::postProcessing(cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Box>& detectedBoxes)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    int counter = 0;
    for (size_t i = 0; i < outs.size(); ++i)
    {   
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                counter += 1;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        detectedBoxes.push_back(Box(box.x, box.y, box.x+box.width, box.y+box.height, confidences[idx]));
    }
}
