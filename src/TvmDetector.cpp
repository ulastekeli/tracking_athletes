#include "TvmDetector.h"
#include <fstream>

TvmDetector::TvmDetector(const std::string& modelPath) 
    : modelPath(modelPath), confThreshold(0.4), nmsThreshold(0.2){
    loadModel();
}
TvmDetector::~TvmDetector() {
    // Cleanup code goes here
}
void TvmDetector::loadModel()
{
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelPath);
    DLDevice dev{kDLCPU, 0};

    // Get the functions from the module.
    mod = mod_syslib.GetFunction("default")(dev);
    set_input = mod.GetFunction("set_input");
    get_output = mod.GetFunction("get_output");
    run_model = mod.GetFunction("run");

}

std::vector<cv::Mat> TvmDetector::ND2CV(tvm::runtime::NDArray ndarray, int w, int h){
    // Get the shape of the array
    tvm::runtime::ShapeTuple shape_tuple = ndarray.Shape();
    
    // Convert the ShapeTuple to a std::vector
    std::vector<int64_t> shape(shape_tuple.begin(), shape_tuple.end());

    // Get the raw data from the array
    float* data = static_cast<float*>(ndarray->data);

    // Create a vector of cv::Mat
    std::vector<cv::Mat> mats(6);

    // For each cv::Mat in the vector
    for(int i = 0; i < 6; ++i)
    {
        // Create a cv::Mat with the data and appropriate shape
        // Note: The data pointer is offset to the start of the data for each mat
        int size[] = {3, shape[2], shape[3]};
        mats[i] = cv::Mat(3, size, CV_32F, data + i*3*shape[2]*shape[3]);
    }
    return mats;
}

std::vector<Box> TvmDetector::run(cv::Mat frame)
{
    DLDevice dev{kDLCPU, 0};
    constexpr int dtype_code = kDLFloat;
    constexpr int dtype_bits = 32;
    constexpr int dtype_lanes = 1;
    constexpr int device_type = kDLCPU;
    constexpr int device_id = 0;
    constexpr int in_ndim = 4;

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(960, 544), cv::Scalar(0,0,0), true, false);
    // frame.convertTo(frame, CV_32F);
    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({1, 3, 544, 960}, DLDataType{kDLFloat, 32, 1}, dev);
    int64_t shape[4] = {1, 3, 544, 960};

    float* data = reinterpret_cast<float*>(frame.data);
    std::memcpy(static_cast<float*>(input->data), data, 3 * 960 * 544 * sizeof(float));
 
    set_input("data", input);
    run_model();

    tvm::runtime::NDArray res = tvm::runtime::NDArray::Empty({1, 18, 68, 120}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray res1 = tvm::runtime::NDArray::Empty({1, 18, 34, 60}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray res2 = tvm::runtime::NDArray::Empty({1, 18, 17, 30}, DLDataType{kDLFloat, 32, 1}, dev);

    int out_ndim = 4;

    get_output(0,res);
    get_output(4,res1);                
    get_output(8,res2);

    std::vector<float> output1(static_cast<float*>(res->data), static_cast<float*>(res->data) + 3*6*68*120);
    std::vector<float> output2(static_cast<float*>(res->data), static_cast<float*>(res->data) + 3*6*34*60);
    std::vector<float> output3(static_cast<float*>(res->data), static_cast<float*>(res->data) + 3*6*17*30);

    std::cout<<output1[213]<<std::endl;
    std::vector<std::vector<cv::Mat>> all_mats;
    all_mats.push_back(ND2CV(res2, 17, 30)); // mats3
    all_mats.push_back(ND2CV(res1, 34, 60)); // mats2
    all_mats.push_back(ND2CV(res, 68, 120));  // mats1


    std::vector<Box> detectedBoxes = postProcessing(frame, all_mats);

    return detectedBoxes;
}

std::vector<Box> TvmDetector::NMS(std::vector<Box>& boxes, float threshold) {
    // Sort the boxes based on the confidence scores
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Box> final_boxes; // This will hold the final boxes after NMS
    while (boxes.size() != 0) {
        // Take the box with the highest score
        Box max_conf_box = boxes[0];
        final_boxes.push_back(max_conf_box);

        // Compare this box with all the other boxes in the list
        std::vector<Box> remaining_boxes;
        for (int i = 1; i < boxes.size(); ++i) {
            if (IOU(max_conf_box, boxes[i]) <= threshold) {
                remaining_boxes.push_back(boxes[i]);
            }
        }

        // Replace the current list of boxes with the remaining boxes after comparison
        boxes = remaining_boxes;
    }

    return final_boxes;
}

float TvmDetector::IOU(const Box& a, const Box& b) {
    // Calculate the coordinates of the intersection rectangle
    float inter_xmin = std::max(a.xmin, b.xmin);
    float inter_ymin = std::max(a.ymin, b.ymin);
    float inter_xmax = std::min(a.xmax, b.xmax);
    float inter_ymax = std::min(a.ymax, b.ymax);

    // Calculate the area of intersection rectangle
    float inter_area = std::max(0.0f, inter_xmax - inter_xmin + 1) * std::max(0.0f, inter_ymax - inter_ymin + 1);

    // Calculate the area of both rectangles
    float a_area = (a.xmax - a.xmin + 1) * (a.ymax - a.ymin + 1);
    float b_area = (b.xmax - b.xmin + 1) * (b.ymax - b.ymin + 1);

    // Calculate the IoU
    float iou = inter_area / (a_area + b_area - inter_area);

    return iou;
}

std::vector<Box> TvmDetector::postProcessing(cv::Mat frame, std::vector<std::vector<cv::Mat>> outs)
{
    std::vector<Box> boxes;
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    for (int out = 0; out < 3; ++out){
        std::vector<cv::Mat> current_yolo_layer = outs[out];
        int cols = current_yolo_layer[0].size[1];
        int rows = current_yolo_layer[0].size[2];
        for (int anchor_no = 0; anchor_no < 3; ++anchor_no){
            for (int i = 0; i < rows; ++i){
                for (int j = 0; j < cols; ++j){
                    float score = current_yolo_layer[4].at<float>(anchor_no, i, j);
                    // std::cout<<score<<std::endl;
                    if (score < confThreshold)
                        continue;
                    float bx = (j + current_yolo_layer[0].at<float>(anchor_no, i, j)) / cols;
                    float by = (i + current_yolo_layer[1].at<float>(anchor_no, i, j)) / rows;
                    float baseW = current_yolo_layer[3].at<float>(anchor_no, i, j);
                    float baseH = current_yolo_layer[2].at<float>(anchor_no, i, j);
                    float expW = exp(baseW);
                    float expH = exp(baseH);
                    float bw = expW * anchors[out*3*2 + anchor_no*2] / frame_width;
                    float bh = expH * anchors[out*3*2 + anchor_no*2 +1] / frame_height;
                    Box box = Box(bx-bw/2, by-bh/2, bx+bw/2, by+bh/2, score);
                    // box.print();
                    boxes.push_back(box);
                }
            }
        }
    }

    boxes = NMS(boxes, nmsThreshold);
    return boxes;
}
