#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;


vector<String> getOutputsNames(cv::dnn::Net net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    for (size_t i = 0; i < names.size(); i++)
    {
        std::cout<<names[i]<<std::endl;
    }
    
    return names;
}


void post_process(const vector<Mat>& outs, float *bbox_to_return, int cols, int rows)
{
    vector<int> classIds;
    vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float max_score = 0;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.4)
            {
                float centerX = (data[0] * cols);
                float centerY = (data[1] * rows);
                float width = (data[2] * cols);
                float height = (data[3] * rows);

                boxes.push_back(cv::Rect(0, 0, 5, 5));                
                
                if((float)confidence > max_score){
                    max_score = (float)confidence;
                    bbox_to_return[0] = centerX;
                    bbox_to_return[1] = centerY;
                    bbox_to_return[2] = width;
                    bbox_to_return[3] = height;
                    bbox_to_return[4] = (float)confidence;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                }
            }
        }
    }
    std::cout<<boxes.size()<<std::endl;
    std::cout<<confidences.size()<<std::endl;
}

int* to_yolo_box(float* output, int* output_box, int h, int w){

    output_box[0] = std::max(0, static_cast<int>(output[0] - (output[2])/2));
    output_box[1] = std::max(0, static_cast<int>(output[1] - (output[3])/2));
    output_box[2] = std::min(w-1, static_cast<int>(output[0] + (output[2])/2));
    output_box[3] = std::min(h-1, static_cast<int>(output[1] + (output[3])/2));

    output_box[3] = output_box[3] - output_box[1];
    output_box[2] = output_box[2] - output_box[0];

    return output_box;
}


class Detector{

    private:
     cv::dnn::Net net;

    public:
        Detector(std::string model_cfg, std::string model_weights){
            this->net = readNetFromDarknet(model_cfg, model_weights);
            this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
            this->net.setPreferableTarget(DNN_TARGET_CPU);
            //this->net = net;
        }

        void forward(cv::Mat frame, float *coordinateBox){

            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(960, 544), cv::Scalar(0,0,0), true, false);
            //Sets the input to the network
            this->net.setInput(blob);
            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            this->net.forward(outs, getOutputsNames(this->net));

            // Remove the bounding boxes with low confidence
            post_process(outs, coordinateBox, frame.cols, frame.rows);
        }

        cv::Mat get_person_img(cv::Mat frame, int *real_box){
            float coordinateBox[5] = {0};
            this->forward(frame, coordinateBox);
            to_yolo_box(coordinateBox, real_box, frame.rows, frame.cols);
            cv::Rect myROI(real_box[0], real_box[1], real_box[2], real_box[3]);
            cv::Mat croppedImage = frame(myROI);
            return croppedImage;
        }
};


int main(){
    std::cout << "app start " << std::endl;

    Detector detector = Detector("/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.cfg", "/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.weights");
    cv::Mat img = cv::imread("/home/rap018/workspace_ulas/misc/assignment/data/example.jpg",1);
    float coordinateBox[5] = {0};
    detector.forward(img, coordinateBox);
    cv::rectangle(img, cv::Rect(cv::Point(coordinateBox[0]-coordinateBox[2]/2, coordinateBox[1]-coordinateBox[3]/2), cv::Point(coordinateBox[0]+coordinateBox[2]/2, coordinateBox[1]+coordinateBox[3]/2)), cv::Scalar(0, 255, 0));
    cv::imwrite("out_old.jpg", img);
    printf("imagenew.bin\n");
    printf("Confidence: %.3f\n",coordinateBox[4]);
    printf("X Center: %.3f\n",coordinateBox[0]);
    printf("Y Center: %.3f\n",coordinateBox[1]);
    printf("Width: %.3f\n",coordinateBox[2]);
    printf("Height: %.3f\n",coordinateBox[3]);   

}
