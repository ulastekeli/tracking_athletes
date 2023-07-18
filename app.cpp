#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Output.H>  // For Fl_Output widget
#include "Tracker.h"
#include "Box.h"
#include "Track.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <thread>
#include <chrono>

void sleepFunction(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

std::string modelWeights = "../models/dev_models/pd_tiny4_best.weights";
std::string modelConfiguration = "../models/dev_models/pd_tiny4.cfg";
std::string reidModelPath = "../models/dep_models/osnet1.so";

std::string detectionRoot = "../output/dets";
std::string imageSaveRoot = "../output/images";
std::string croppedSaveRoot = "../output/cropped";

// Global color vector
std::vector<cv::Scalar> colors;
cv::Mat bgImg;

std::string selectedFilePath = "../data/clip_tennis.mp4";  // To store the selected file path
std::string modelOption = "tennis";
std::string inferenceOption = "tracking";
int fpsOption = 30;  // Default value
bool showOption = true;
bool saveOption = true;

class VideoProcessor
{
public:
    VideoProcessor(Fl_Box *box) : box(box) {}
    void processVideo(std::string videoPath, std::string inferenceOption);
    void displayTracks(std::string videoPath);
    void getCroppedImages(std::string videoPath);

private:
    Fl_Box *box;
};


void generateColors(int numColors)
{
    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, 255);

    // Generate the colors
    colors.resize(numColors);
    for (int i = 0; i < numColors; ++i)
    {
        int r = uni(rng);
        int g = uni(rng);
        int b = uni(rng);
        colors[i] = cv::Scalar(b, g, r); // OpenCV uses BGR color order
    }
}

cv::Scalar getIdColor(int id)
{
    // Return the color corresponding to the id
    // (Use modulo to prevent out-of-bounds access)
    return colors[id % colors.size()];
}

void createDirectory(const std::string &dirName)
{
    std::filesystem::path dirPath(dirName);
    if (!std::filesystem::exists(dirPath))
    {
        std::filesystem::create_directory(dirPath);
        std::cout << "Directory created: " << dirPath << std::endl;
    }
    else
    {
        std::cout << "Directory already exists: " << dirPath << std::endl;
    }
}

void buttonCb(Fl_Widget* btn, void* userdata) {
    Fl_Native_File_Chooser chooser;
    chooser.title("Pick a file");
    chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.filter("Video\t*.{avi,mkv,mp4}");
    chooser.directory("../data"); // replace with your directory path

    // Update modelWeights and modelConfiguration based on model_option
    if (modelOption == "football") {
        modelWeights = "../models/dev_models/pd_tiny4_best.weights";
        modelConfiguration = "../models/dev_models/pd_tiny4.cfg";
    } else if (modelOption == "tennis") {
        modelWeights = "../models/dev_models/pd_tiny4_tennis.weights";
        modelConfiguration = "../models/dev_models/pd_tiny4_tennis.cfg";
    }

    if (chooser.show() == 0)
    {
        selectedFilePath = chooser.filename();
        Fl_Output* output = (Fl_Output*)userdata;
        output->value(selectedFilePath.c_str());
    }

}

// Callback for FPS input
void fpsInputCb(Fl_Widget* w, void* userdata) {
    Fl_Int_Input* input = (Fl_Int_Input*)w;
    fpsOption = std::stoi(input->value());
}

void displayCb(Fl_Widget* btn, void* userdata) {
    VideoProcessor *processor = static_cast<VideoProcessor *>(userdata);
    processor->displayTracks(selectedFilePath);
}

void startCb(Fl_Widget* btn, void* userdata) {
    VideoProcessor *processor = static_cast<VideoProcessor *>(userdata);
    processor->processVideo(selectedFilePath, inferenceOption);
}

// Callback for model choice
void modelChoiceCb(Fl_Widget* w, void* userdata) {
    Fl_Choice* choice = (Fl_Choice*)w;
    modelOption = choice->text();
}

// Callback for inference choice
void inferenceChoiceCb(Fl_Widget* w, void* userdata) {
    Fl_Choice* choice = (Fl_Choice*)w;
    inferenceOption = choice->text();
}

// Callback for when the check button is clicked
void checkCb(Fl_Widget* btn, void* userdata) {
    Fl_Check_Button* check = static_cast<Fl_Check_Button*>(btn);
    showOption = check->value();
}

Fl_RGB_Image *copyMat2FlImage(const cv::Mat &mat)
{
    cv::Mat matConverted;
    cv::cvtColor(mat, matConverted, cv::COLOR_BGR2RGB);
    uchar *buffer = new uchar[matConverted.rows * matConverted.cols * matConverted.channels()];
    std::memcpy(buffer, matConverted.data, matConverted.rows * matConverted.cols * matConverted.channels());
    return new Fl_RGB_Image(buffer, matConverted.cols, matConverted.rows, 3);
}

void VideoProcessor::getCroppedImages(std::string videoPath)
{
    generateColors(100);
    std::filesystem::path path(videoPath);
    std::string videoName = path.stem();

    // Append the video filename to the detectionPath
    std::string videoDetectionPath = detectionRoot + "/" + videoName;
    std::string croppedSavePath = croppedSaveRoot + "/" + videoName;

    createDirectory(croppedSavePath);
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Can't open video file" << std::endl;
        return;
    }
    else
    {
        std::cout << "Cap is opened" << std::endl;
    }

    int frameId = 0;
    cv::Mat frame;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
            break;
        std::cout << "Displaying frame " << frameId << std::endl;
        // Open the corresponding text file
        std::ifstream inFile(videoDetectionPath + "/" + std::to_string(frameId) + ".txt");
        if (!inFile)
        {
            std::cerr << "ERROR: Can't open file " << videoDetectionPath + "/" + std::to_string(frameId) + ".txt" << std::endl;
            return;
        }

        // Read each line of the file
        std::string line;
        while (std::getline(inFile, line))
        {
            std::istringstream iss(line);
            int id, xmin, ymin, xmax, ymax;

            if (!(iss >> id >> xmin >> ymin >> xmax >> ymax))
            {
                std::cerr << "ERROR: Can't parse line " << line << " in file " << frameId << ".txt" << std::endl;
                return;
            }

            cv::Rect roi(xmin, ymin, xmax - xmin, ymax - ymin);                                                          // Define the region of interest
            cv::Mat cropped = frame(roi);                                                                                // Crop the image
            cv::imwrite(croppedSaveRoot + "/" + std::to_string(frameId) + "_" + std::to_string(id) + ".png", cropped); // Save the cropped image
        }

        frameId++;
    }

    cap.release();
}

void VideoProcessor::displayTracks(std::string videoPath)
{
    std::filesystem::path path(videoPath);
    std::string videoName = path.stem();

    // Append the video filename to the detectionPath
    std::string videoDetectionPath = detectionRoot + "/" + videoName;
    std::string imageSavePath = imageSaveRoot + "/" + videoName;
    createDirectory(imageSavePath);
    cv::VideoCapture cap(videoPath);
    int intervalMs = static_cast<int>(1000. / fpsOption);

    int frameId = 0;
    cv::Mat frame;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
            break;

        std::ifstream inFile(videoDetectionPath + "/" + std::to_string(frameId) + ".txt");
        if (!inFile)
        {
            std::cerr << "ERROR: Can't open file " << videoDetectionPath + "/" + std::to_string(frameId) + ".txt" << std::endl;
            return;
        }

        std::string line;
        while (std::getline(inFile, line))
        {
            std::istringstream iss(line);
            int id, xmin, ymin, xmax, ymax;

            if (!(iss >> id >> xmin >> ymin >> xmax >> ymax))
            {
                iss.clear(); // Clear the error state
                iss.str(line);
                if (!(iss >> xmin >> ymin >> xmax >> ymax)){
                    std::cerr << "ERROR: Can't parse line " << line << " in file " << frameId << ".txt" << std::endl;
                    return;
                }
                id = -1;
            }
            if(id != -1){
                cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), getIdColor(id), 2);
                std::string label = std::to_string(id);
                int baseline;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
                cv::putText(frame, label, cv::Point(xmin, ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 1, getIdColor(id), 2);
            }
            else{
                cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), getIdColor(0), 2);
            }
        }
        if (showOption)
        {
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(box->w(), box->h()));
            Fl_RGB_Image *newImage = copyMat2FlImage(resized);
            delete (Fl_RGB_Image *)box->image();
            box->image(newImage);
            box->redraw();
            Fl::check();
        }
        else
        {
            cv::imwrite(imageSavePath + "/" + std::to_string(frameId) + ".jpg", frame);
        }

        frameId++;
        sleepFunction(intervalMs);
    }

    cap.release();
}

void VideoProcessor::processVideo(std::string videoPath, std::string inferenceOption)
{
    std::cout<<inferenceOption<<std::endl;
    std::cout<<modelWeights<<std::endl;
    std::cout<<videoPath<<std::endl;

    sleepFunction(1000);
    ObjectDetector detector(modelWeights, modelConfiguration);
    Tracker tracker(reidModelPath);
    cv::VideoCapture cap(videoPath);

    // Extract the video filename from the path
    std::filesystem::path path(videoPath);
    std::string videoName = path.stem();

    // Append the video filename to the detectionPath
    std::string videoDetectionPath = detectionRoot + "/" + videoName;
    std::string imageSavePath = imageSaveRoot + "/" + videoName;

    createDirectory(videoDetectionPath);
    createDirectory(imageSavePath);

    int frameNumber = 0;
    cv::Mat frame;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<Box> detectedObjects = detector.detectObjects(frame);
        tracker.match(detectedObjects, frame);

        std::ofstream outFile;
        std::stringstream ss;
        ss << videoDetectionPath << "/" << frameNumber << ".txt";
        outFile.open(ss.str());

        // Distinguish between detection and tracking results based on inferenceOption
        if (inferenceOption == "detection only") {
            for(const auto& box : detectedObjects) {
                outFile << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
                if (showOption || saveOption) {
                    cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(255, 0, 0), 2);
                }
            }
        } else if (inferenceOption == "tracking") {
            for (const auto &track : tracker.getLastFrameTracks())
            {
                const auto &box = track.box_history.back();
                outFile << track.id << " " << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
                if (showOption || saveOption)
                {
                    cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), getIdColor(track.id), 2);

                    std::string label = std::to_string(track.id);
                    int baseline;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
                    cv::putText(frame, label, cv::Point(box.xmin, box.ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 1, getIdColor(track.id), 2);
                }
            }
        }

        outFile.close();
        if (showOption)
        {
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(box->w(), box->h()));
            Fl_RGB_Image *newImage = copyMat2FlImage(resized);
            delete (Fl_RGB_Image *)box->image();
            box->image(newImage);
            box->redraw();
            Fl::check();
        }
        if (saveOption)
        {
            cv::imwrite(imageSavePath + "/" + std::to_string(frameNumber) + ".jpg", frame);
        }
        frameNumber++;
    }
    cap.release();
    // Fl_RGB_Image *tmp = copyMat2FlImage(bgImg);
    // delete (Fl_RGB_Image *)box->image();
    // box->image(tmp);
    // box->redraw();
    // Fl::check();
}

int main(int argc, char **argv)
{
    generateColors(100);

    Fl_Window *window = new Fl_Window(1800, 1000, "Track Athletes");

    Fl_Button *button = new Fl_Button(80, 20, 200, 50, "Choose video");

    Fl_Output* output = new Fl_Output(80, 80, 200, 25);  // New output widget to display the selected file path

    Fl_Choice* modelChoice = new Fl_Choice(80, 140, 200, 25, "Model:");
    modelChoice->add("football model");
    modelChoice->add("tennis model");
    modelChoice->value(1);
    modelChoice->callback(modelChoiceCb);
    modelOption = "tennis";
    

    Fl_Choice* inferenceChoice = new Fl_Choice(80, 200, 200, 25, "Inference:");
    inferenceChoice->add("detection only");
    inferenceChoice->add("tracking");
    inferenceChoice->value(1);  // Set the default choice to "tracking"
    inferenceChoice->callback(inferenceChoiceCb);
    inferenceOption = "tracking";

    Fl_Check_Button *check = new Fl_Check_Button(80, 240, 300, 50, "Show while processing");
    check->value(1);  // Set the initial state to ticked
    check->callback(checkCb, nullptr);
    
    // Add an FPS input field
    Fl_Int_Input* fpsInput = new Fl_Int_Input(80, 300, 200, 25, "FPS:");
    fpsInput->value(std::to_string(fpsOption).c_str());  // Set the default value to 30
    fpsInput->callback(fpsInputCb);

    Fl_Button *displayButton = new Fl_Button(80, 340, 200, 50, "Display Detections/Tracks");  // New start button
    Fl_Button *startButton = new Fl_Button(80, 400, 200, 50, "Start");  // New start button


    Fl_Box *box = new Fl_Box(400, 20, 1360, 960);
    cv::Mat img = cv::imread("../data/example.jpg");
    cv::resize(img, img, cv::Size(1360, 940));
    Fl_RGB_Image *tmp = copyMat2FlImage(img);
    box->image(tmp);

    VideoProcessor processor(box);
    button->callback(buttonCb, output);
    displayButton->callback(displayCb, &processor); 
    startButton->callback(startCb, &processor); 

    window->end();
    window->show(argc, argv);

    return Fl::run();
}
