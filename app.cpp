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
#include <filesystem>
#include <random>

std::string modelWeights = "../models/dev_models/pd_tiny4_tennis.weights";
std::string modelConfiguration = "../models/dev_models/pd_tiny4_tennis.cfg";
std::string reid_model_path = "../models/dep_models/osnet1.so";
std::string filePath = "../data/clip.mp4";
std::string detectionPath = "../output/dets";
std::string imgSavePath = "../output/images";
bool show = false;
bool save = true;

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


void getCroppedImages(std::string videoPath) {
    generateColors(100);
    create_directory("../output/cropped");
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Can't open video file" << std::endl;
        return;
    }
    else{
        std::cout<<"Cap is opened"<<std::endl;
    }

    int frameId = 0;
    cv::Mat frame;

    while(1) {
        cap >> frame;
        if (frame.empty()) break;
        std::cout<<"Displaying frame "<<frameId<<std::endl;
        // Open the corresponding text file
        std::ifstream inFile(detectionPath+"/"+ std::to_string(frameId) + ".txt");
        if (!inFile) {
            std::cerr << "ERROR: Can't open file " << detectionPath+"/"+ std::to_string(frameId) + ".txt" << std::endl;
            return;
        }

        // Read each line of the file
        std::string line;
        while (std::getline(inFile, line)) {
            std::istringstream iss(line);
            int id, xmin, ymin, xmax, ymax;

            if (!(iss >> id >> xmin >> ymin >> xmax >> ymax)) {
                std::cerr << "ERROR: Can't parse line " << line << " in file " << frameId << ".txt" << std::endl;
                return;
            }

            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin); // Define the region of interest
            cv::Mat cropped = frame(roi); // Crop the image
            cv::imwrite("../output/cropped/" + std::to_string(frameId) + "_" + std::to_string(id) + ".png", cropped); // Save the cropped image

        }

        frameId++;
    }

    cap.release();
}


void displayTracks(std::string videoPath) {
    generateColors(100);
    create_directory(imgSavePath);
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Can't open video file" << std::endl;
        return;
    }
    else{
        std::cout<<"Cap is opened"<<std::endl;
    }

    int frameId = 0;
    cv::Mat frame;

    while(1) {
        cap >> frame;
        if (frame.empty()) break;
        std::cout<<"Displaying frame "<<frameId<<std::endl;
        // Open the corresponding text file
        std::ifstream inFile(detectionPath+"/"+ std::to_string(frameId) + ".txt");
        if (!inFile) {
            std::cerr << "ERROR: Can't open file " << detectionPath+"/"+ std::to_string(frameId) + ".txt" << std::endl;
            return;
        }

        // Read each line of the file
        std::string line;
        while (std::getline(inFile, line)) {
            std::istringstream iss(line);
            int id, xmin, ymin, xmax, ymax;

            if (!(iss >> id >> xmin >> ymin >> xmax >> ymax)) {
                std::cerr << "ERROR: Can't parse line " << line << " in file " << frameId << ".txt" << std::endl;
                return;
            }

            // Draw rectangle for bounding box
            cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), getIdColor(id), 2);

            // Draw text for track id
            std::string label = std::to_string(id);
            int baseline;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
            cv::putText(frame, label, cv::Point(xmin, ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 1, getIdColor(id), 2);
        }

        if(show){
            // Display the image
            cv::imshow("Image", frame);
            int key = cv::waitKey(1);  // Wait for 30 ms

            // Exit if ESC key is pressed
            if (key == 27) {
                break;
            }
        }else{
            cv::imwrite(imgSavePath+"/"+ std::to_string(frameId) + ".jpg", frame);
        }

        frameId++;
    }

    cap.release();
    if(show)
        cv::destroyAllWindows();
}


void process_video(std::string video_path) {
    ObjectDetector detector(modelWeights, modelConfiguration);
    std::cout<< " Object detector created " <<std::endl;
    Tracker tracker(reid_model_path);
    std::cout<< " Tracker created " <<std::endl;
    cv::VideoCapture cap(video_path); 

    if(!cap.isOpened()) {
        std::cout << "Error opening video file" << std::endl;
        return;
    }else{
        std::cout<< " Cap opened " <<std::endl;
    }

    create_directory(detectionPath);
    create_directory(imgSavePath);

    int frameNumber = 0;
    int track_frame_no = -1;
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
        ss << detectionPath << "/" << frameNumber << ".txt";
        outFile.open(ss.str());
        for(const auto& track : tracker.getTracks()) {
            track_frame_no = track.frame_history.back();
            if (track_frame_no != frameNumber){
                continue;
            }
            const auto& box = track.box_history.back();
            outFile << track.id << " " << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
            if (show || save){
                // Draw rectangle for bounding box
                cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(0, 255, 0), 2);

                // Draw text for track id
                std::string label = std::to_string(track.id);
                int baseline;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::putText(frame, label, cv::Point(box.xmin, box.ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
        outFile.close();
        if (show){
            // Display the image with bounding boxes
            cv::imshow("Image", frame);
            cv::waitKey(1);  // Wait for key press to close window
        }
        if (save){
            cv::imwrite(imgSavePath +"/"+ std::to_string(frameNumber) + ".jpg", frame);
        }        
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
    // std::cout << cv::getBuildInformation() << std::endl;
    std::filesystem::path file_path(filePath);
    if(!std::filesystem::exists(file_path)){
        std::cout << filePath << " does not exist" <<std::endl;
    }
    process_video(filePath);
    // displayTracks(filePath);
    getCroppedImages(filePath);
}
