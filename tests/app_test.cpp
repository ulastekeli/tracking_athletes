#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Check_Button.H>
#include "Tracker.h"
#include "Box.h"
#include "Track.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>

std::string modelWeights = "../models/dev_models/pd_tiny4_tennis.weights";
std::string modelConfiguration = "../models/dev_models/pd_tiny4_tennis.cfg";
std::string reid_model_path = "../models/dep_models/osnet1.so";
std::string detectionPath = "../output/dets";
std::string imgSavePath = "../output/images";
bool show = true;
bool save = true;


class VideoProcessor {
public:
    VideoProcessor(Fl_Box* box) : box(box) {}
    void process_video(std::string video_path);

private:
    Fl_Box* box;
};


// Global color vector
std::vector<cv::Scalar> colors;

void generateColors(int num_colors) {
    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, 255);

    // Generate the colors
    colors.resize(num_colors);
    for (int i = 0; i < num_colors; ++i) {
        int r = uni(rng);
        int g = uni(rng);
        int b = uni(rng);
        colors[i] = cv::Scalar(b, g, r);  // OpenCV uses BGR color order
    }
}

cv::Scalar getIdColor(int id) {
    // Return the color corresponding to the id
    // (Use modulo to prevent out-of-bounds access)
    return colors[id % colors.size()];
}

void create_directory(const std::string& dir_name) {
    std::filesystem::path dir_path(dir_name);
    if(!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directory(dir_path);
        std::cout << "Directory created: " << dir_path << std::endl;
    } else {
        std::cout << "Directory already exists: " << dir_path << std::endl;
    }
}

// Callback for when the button is clicked
void button_cb(Fl_Widget* btn, void* userdata) {
    VideoProcessor* processor = static_cast<VideoProcessor*>(userdata);
    Fl_Native_File_Chooser chooser;
    chooser.title("Pick a file");
    chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.filter("Video\t*.{avi,mkv,mp4}");

    if (chooser.show() == 0) {
        const char* filename = chooser.filename();
        processor->process_video(filename);
    }
}


Fl_RGB_Image* CopyMat2FlImage(const cv::Mat& mat) {
    cv::Mat mat_converted;
    cv::cvtColor(mat, mat_converted, cv::COLOR_BGR2RGB);
    uchar* buffer = new uchar[mat_converted.rows * mat_converted.cols * mat_converted.channels()];
    std::memcpy(buffer, mat_converted.data, mat_converted.rows * mat_converted.cols * mat_converted.channels());
    return new Fl_RGB_Image(buffer, mat_converted.cols, mat_converted.rows, 3);
}

void VideoProcessor::process_video(std::string video_path) {
    ObjectDetector detector(modelWeights, modelConfiguration);
    Tracker tracker(reid_model_path);
    cv::VideoCapture cap(video_path); 

    create_directory(detectionPath);
    create_directory(imgSavePath);

    int frameNumber = 0;
    cv::Mat frame;

    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<Box> detectedObjects = detector.detectObjects(frame);
        tracker.match(detectedObjects, frame);

        std::ofstream outFile;
        std::stringstream ss;
        ss << detectionPath << "/" << frameNumber << ".txt";
        outFile.open(ss.str());
        for(const auto& track : tracker.getTracks()) {
            const auto& box = track.box_history.back();
            outFile << track.id << " " << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
            if (show || save){
                cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), getIdColor(track.id), 2);

                std::string label = std::to_string(track.id);
                int baseline;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
                cv::putText(frame, label, cv::Point(box.xmin, box.ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 1, getIdColor(track.id), 2);
            }
        }
        outFile.close();
        if (show){
            cv::resize(frame, frame, cv::Size(box->w(), box->h()));
            Fl_RGB_Image* newImage = CopyMat2FlImage(frame);
            delete (Fl_RGB_Image*)box->image();
            box->image(newImage);
            box->redraw();
            Fl::check();
        }
        if (save){
            cv::imwrite(imgSavePath +"/"+ std::to_string(frameNumber) + ".jpg", frame);
        }        
        frameNumber++;
    }
    cap.release();
}

int main(int argc, char** argv) {
    generateColors(100);

    Fl_Window* window = new Fl_Window(1800, 1060, "Track Athletes");

    Fl_Button* button = new Fl_Button(20, 20, 200, 50, "Button");

    Fl_Check_Button* check = new Fl_Check_Button(20, 100, 300, 50, "Show while processing");
    
    Fl_Box* box = new Fl_Box(400, 20, 1360, 1020);
    cv::Mat img = cv::imread("../data/example.jpg");
    cv::resize(img, img, cv::Size(1360, 1020));
    Fl_RGB_Image* tmp = CopyMat2FlImage(img);
    box->image(tmp);

    VideoProcessor processor(box);
    button->callback(button_cb, &processor);

    window->end();
    window->show(argc, argv);

    return Fl::run();
}

