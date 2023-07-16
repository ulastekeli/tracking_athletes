#ifndef REID_MODEL_H
#define REID_MODEL_H

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>

#include <opencv2/opencv.hpp>

class ReidModel {
public:
    ReidModel(const std::string& model_path);
    ~ReidModel();
    
    std::vector<float> run(const cv::Mat& image);

private:
    tvm::runtime::Module mod;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run_model;
};

#endif // REID_MODEL_H
