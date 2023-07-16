#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_JPEG_Image.H>
#include "Detector.h"
#include <iostream>
#include <fstream>
#include <sstream>

std::string modelWeights = "../models/dev_models/pd.weights";
std::string modelConfiguration = "../models/dev_models/pd.cfg";
std::string classNamesFile = "../models/dev_models/pd.names";

void process_video(std::string video_path) {
    ObjectDetector detector(modelWeights, modelConfiguration, classNamesFile);
    cv::VideoCapture cap(video_path); 

    if(!cap.isOpened()) {
        std::cout << "Error opening video file" << std::endl;
        return;
    }

    int frameNumber = 0;
    cv::Mat frame;

    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<Box> detectedBoxes = detector.detectObjects(frame);
        std::cout << "Processed frame no :" << frameNumber << std::endl;

        std::ofstream outFile;
        std::stringstream ss;
        ss << "output/detections_" << frameNumber << ".txt";
        outFile.open(ss.str());

        for(auto& box : detectedBoxes) {
            outFile << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << " " << box.confidence << std::endl;
        }

        outFile.close();
        frameNumber++;
    }
    cap.release();
}

// Callback for when the button is clicked
void button_cb(Fl_Widget* btn, void* userdata) {
    Fl_Native_File_Chooser chooser;
    chooser.title("Pick a file");
    chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.filter("Video\t*.{avi,mkv,mp4}");

    if (chooser.show() == 0) {
        // File was chosen
        const char* filename = chooser.filename();
        process_video(filename);
    }
}

int main(int argc, char** argv) {
    Fl_Window* window = new Fl_Window(400, 300);
    Fl_Button* button = new Fl_Button(50, 50, 100, 25, "Choose File");
    button->callback(button_cb);
    Fl_Box* box = new Fl_Box(50, 100, 200, 200);
    Fl_JPEG_Image* img = new Fl_JPEG_Image("../data/example.jpg");
    box->image(img);
    window->end();
    window->show(argc, argv);
    return Fl::run();
}
