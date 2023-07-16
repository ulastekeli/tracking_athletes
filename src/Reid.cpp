#include "Reid.h"

ReidModel::ReidModel(const std::string& model_path)
{
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_path);
    DLDevice dev{kDLCPU, 0};

    // Get the functions from the module.
    tvm::runtime::Module gmod = mod_syslib.GetFunction("default")(dev);
    set_input = gmod.GetFunction("set_input");
    get_output = gmod.GetFunction("get_output");
    run_model = gmod.GetFunction("run");
}

ReidModel::~ReidModel()
{
    // Destructor. No need to do anything as the Module's destructor will be called automatically.
}

std::vector<float> ReidModel::run(const cv::Mat& image)
{
    DLDevice dev{kDLCPU, 0};
    // Assumes image is in BGR format, CV_32F type, and already resized to the correct dimensions.
    cv::Mat image_transposed;
    cv::transpose(image, image_transposed);    
    std::cout<<image_transposed.cols<<std::endl;
    std::cout<<image_transposed.rows<<std::endl;
    DLTensor* input;
    constexpr int dtype_code = kDLFloat;
    constexpr int dtype_bits = 32;
    constexpr int dtype_lanes = 1;
    constexpr int device_type = kDLCPU;
    constexpr int device_id = 0;
    constexpr int in_ndim = 4;
    int64_t shape[4] = {1, 3, 256, 128};
    // std::cout<<"Before tvmarrayalloc"<<std::endl;
    TVMArrayAlloc(shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
    // std::cout<<"Before memcpy"<<std::endl;
    TVMArrayCopyFromBytes(input, image_transposed.data, 3 * 256 * 128 * sizeof(float));

    // std::cout << "First element of image_transposed.data: " << image_transposed.at<float>(0,0) << std::endl;
    // std::cout << "First element of x->data: " << static_cast<float*>(input->data)[0] << std::endl;

    // tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 3, 256,128}, DLDataType{kDLFloat, 32, 1}, dev);
    // tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 512}, DLDataType{kDLFloat, 32, 1}, dev);


    // std::cout<<"Before set input"<<std::endl;
    // std::cout << "x pointer: " << x << std::endl;
    // std::cout << "set_input function pointer: " << &set_input << std::endl;
    set_input("input0", input);
    // std::cout<<"Before run model"<<std::endl;

    // Run the model.
    run_model();
    // std::cout<<"After run model"<<std::endl;

    // Get the output.
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 512}, DLDataType{kDLFloat, 32, 1}, dev);
    get_output(0, y);
    // std::cout<<"After get output"<<std::endl;

    std::vector<float> output(static_cast<float*>(y->data), static_cast<float*>(y->data) + 128);
    TVMArrayFree(input);

    return output;
}
