#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_JPEG_Image.H>
#include "Tracker.h"
#include "Box.h"
#include "Track.h"
#include <iostream>
#include <fstream>
#include <sstream>

std::string modelWeights = "../models/dev_models/pd_tiny4.weights";
std::string modelConfiguration = "../models/dev_models/pd_tiny4.cfg";
std::string classNamesFile = "../models/dev_models/pd.names";
std::string reid_model_path = "../models/dep_models/osnet1.so";

void process_video(std::string video_path) {
    ObjectDetector detector(modelWeights, modelConfiguration, classNamesFile);
    std::cout<< " Object detector created " <<std::endl;
    Tracker tracker(reid_model_path);
    std::cout<< " Tracker created " <<std::endl;
    cv::VideoCapture cap(video_path); 
    std::cout<< " Cap opened " <<std::endl;

    if(!cap.isOpened()) {
        std::cout << "Error opening video file" << std::endl;
        return;
    }

    int frameNumber = 0;
    cv::Mat frame;

    while(1) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<Box> detectedObjects = detector.detectObjects(frame);
        std::cout<< " detector ran. no of boxes : " << detectedObjects.size() << std::endl;
        
        tracker.match(detectedObjects, frame);

        std::cout << "Processed frame no :" << frameNumber << std::endl;

        std::ofstream outFile;
        std::stringstream ss;
        ss << "output/tracks_" << frameNumber << ".txt";
        outFile.open(ss.str());

        for(const auto& track : tracker.getTracks()) {
            const auto& box = track.box_history.back();
            outFile << track.id << " " << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
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

// int main(int argc, char** argv) {
//     Fl_Window* window = new Fl_Window(400, 300);
//     Fl_Button* button = new Fl_Button(50, 50, 100, 25, "Choose File");
//     button->callback(button_cb);
//     Fl_Box* box = new Fl_Box(50, 100, 200, 200);
//     Fl_JPEG_Image* img = new Fl_JPEG_Image("../data/example.jpg");
//     box->image(img);
//     window->end();
//     window->show(argc, argv);
//     return Fl::run();
// }
int main(int argc, char** argv) {
    process_video("../data/clip.mp4");
}
