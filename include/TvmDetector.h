#ifndef TVM_DETECTOR_H
#define REID_MOTVM_DETECTOR_HDEL_H

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>
#include "Box.h"

#include <opencv2/opencv.hpp>

class TvmDetector {
public:
    TvmDetector(const std::string& modelPath);
    ~TvmDetector();
    
    std::vector<Box> run(cv::Mat frame);
    std::vector<Box> NMS(std::vector<Box>& boxes, float threshold);
    std::vector<cv::Mat> ND2CV(tvm::runtime::NDArray res2, int w, int h);
    std::vector<Box> postProcessing(cv::Mat frame, std::vector<std::vector<cv::Mat>> outs);
    static float IOU(const Box& a, const Box& b);
    void loadModel();

private:
    std::string modelPath;
    tvm::runtime::Module mod;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run_model;
    float confThreshold;
    float nmsThreshold;
    int anchors[18] = {10, 24,  15, 35,  17, 49,  
                    22, 41,  23, 59,  30, 49,  
                    31, 71,  39, 58,  45, 84};
};

#endif // TVM_DETECTOR_H
