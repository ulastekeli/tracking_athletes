#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_PNG_Image.H>
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


void sleep(int seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}


std::string modelWeights = "../models/dev_models/pd_tiny4_best.weights";
std::string modelConfiguration = "../models/dev_models/pd_tiny4.cfg";
std::string reid_model_path = "../models/dep_models/osnet1.so";

std::string detectionRoot = "../output/dets";
std::string image_save_root = "../output/images";
std::string cropped_save_root = "../output/cropped";

// Global color vector
std::vector<cv::Scalar> colors;
cv::Mat bg_img;

std::string selected_file_path = "../data/clip_tennis.mp4";  // To store the selected file path
std::string model_option = "tennis";
std::string inference_option = "tracking";
bool show = true;
bool save = true;

class VideoProcessor
{
public:
    VideoProcessor(Fl_Box *box) : box(box) {}
    void process_video(std::string video_path, std::string inference_option);
    void displayTracks(std::string videoPath);
    void getCroppedImages(std::string videoPath);

private:
    Fl_Box *box;
};


void generateColors(int num_colors)
{
    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, 255);

    // Generate the colors
    colors.resize(num_colors);
    for (int i = 0; i < num_colors; ++i)
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

void create_directory(const std::string &dir_name)
{
    std::filesystem::path dir_path(dir_name);
    if (!std::filesystem::exists(dir_path))
    {
        std::filesystem::create_directory(dir_path);
        std::cout << "Directory created: " << dir_path << std::endl;
    }
    else
    {
        std::cout << "Directory already exists: " << dir_path << std::endl;
    }
}

void button_cb(Fl_Widget* btn, void* userdata) {
    Fl_Native_File_Chooser chooser;
    chooser.title("Pick a file");
    chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.filter("Video\t*.{avi,mkv,mp4}");
    chooser.directory("../data"); // replace with your directory path

    // Update modelWeights and modelConfiguration based on model_option
    if (model_option == "football") {
        modelWeights = "../models/dev_models/pd_tiny4_best.weights";
        modelConfiguration = "../models/dev_models/pd_tiny4.cfg";
    } else if (model_option == "tennis") {
        modelWeights = "../models/dev_models/pd_tiny4_tennis.weights";
        modelConfiguration = "../models/dev_models/pd_tiny4_tennis.cfg";
    }

    if (chooser.show() == 0)
    {
        selected_file_path = chooser.filename();
        Fl_Output* output = (Fl_Output*)userdata;
        output->value(selected_file_path.c_str());
    }

}

void start_cb(Fl_Widget* btn, void* userdata) {
    VideoProcessor *processor = static_cast<VideoProcessor *>(userdata);
    processor->process_video(selected_file_path, inference_option);
}

// Callback for model choice
void model_choice_cb(Fl_Widget* w, void* userdata) {
    Fl_Choice* choice = (Fl_Choice*)w;
    model_option = choice->text();
}

// Callback for inference choice
void inference_choice_cb(Fl_Widget* w, void* userdata) {
    Fl_Choice* choice = (Fl_Choice*)w;
    inference_option = choice->text();
}

// Callback for when the check button is clicked
void check_cb(Fl_Widget* btn, void* userdata) {
    Fl_Check_Button* check = static_cast<Fl_Check_Button*>(btn);
    show = check->value();
}

Fl_RGB_Image *CopyMat2FlImage(const cv::Mat &mat)
{
    cv::Mat mat_converted;
    cv::cvtColor(mat, mat_converted, cv::COLOR_BGR2RGB);
    uchar *buffer = new uchar[mat_converted.rows * mat_converted.cols * mat_converted.channels()];
    std::memcpy(buffer, mat_converted.data, mat_converted.rows * mat_converted.cols * mat_converted.channels());
    return new Fl_RGB_Image(buffer, mat_converted.cols, mat_converted.rows, 3);
}

void VideoProcessor::getCroppedImages(std::string videoPath)
{
    generateColors(100);
    std::filesystem::path path(videoPath);
    std::string video_name = path.stem();

    // Append the video filename to the detectionPath
    std::string video_detection_path = detectionRoot + "/" + video_name;
    std::string cropped_save_path = cropped_save_root + "/" + video_name;

    create_directory(cropped_save_path);
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
        std::ifstream inFile(video_detection_path + "/" + std::to_string(frameId) + ".txt");
        if (!inFile)
        {
            std::cerr << "ERROR: Can't open file " << video_detection_path + "/" + std::to_string(frameId) + ".txt" << std::endl;
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
            cv::imwrite(cropped_save_root + "/" + std::to_string(frameId) + "_" + std::to_string(id) + ".png", cropped); // Save the cropped image
        }

        frameId++;
    }

    cap.release();
}

void VideoProcessor::displayTracks(std::string videoPath)
{
    std::filesystem::path path(videoPath);
    std::string video_name = path.stem();

    // Append the video filename to the detectionPath
    std::string video_detection_path = detectionRoot + "/" + video_name;
    std::string image_save_path = image_save_root + "/" + video_name;
    create_directory(image_save_path);
    cv::VideoCapture cap(videoPath);

    int frameId = 0;
    cv::Mat frame;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
            break;

        std::ifstream inFile(video_detection_path + "/" + std::to_string(frameId) + ".txt");
        if (!inFile)
        {
            std::cerr << "ERROR: Can't open file " << video_detection_path + "/" + std::to_string(frameId) + ".txt" << std::endl;
            return;
        }

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

            cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), getIdColor(id), 2);

            std::string label = std::to_string(id);
            int baseline;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
            cv::putText(frame, label, cv::Point(xmin, ymin + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 1, getIdColor(id), 2);
        }

        if (show)
        {
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(box->w(), box->h()));
            Fl_RGB_Image *newImage = CopyMat2FlImage(resized);
            delete (Fl_RGB_Image *)box->image();
            box->image(newImage);
            box->redraw();
            Fl::check();
        }
        else
        {
            cv::imwrite(image_save_path + "/" + std::to_string(frameId) + ".jpg", frame);
        }

        frameId++;
    }

    cap.release();
}

void VideoProcessor::process_video(std::string video_path, std::string inference_option)
{
    std::cout<<inference_option<<std::endl;
    std::cout<<modelWeights<<std::endl;
    std::cout<<video_path<<std::endl;

    sleep(2);
    ObjectDetector detector(modelWeights, modelConfiguration);
    Tracker tracker(reid_model_path);
    cv::VideoCapture cap(video_path);

    // Extract the video filename from the path
    std::filesystem::path path(video_path);
    std::string video_name = path.stem();

    // Append the video filename to the detectionPath
    std::string video_detection_path = detectionRoot + "/" + video_name;
    std::string image_save_path = image_save_root + "/" + video_name;

    create_directory(video_detection_path);
    create_directory(image_save_path);

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
        ss << video_detection_path << "/" << frameNumber << ".txt";
        outFile.open(ss.str());

        // Distinguish between detection and tracking results based on inference_option
        if (inference_option == "detection only") {
            for(const auto& box : detectedObjects) {
                outFile << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
                if (show || save) {
                    cv::rectangle(frame, cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax), cv::Scalar(255, 0, 0), 2);
                }
            }
        } else if (inference_option == "tracking") {
            for (const auto &track : tracker.getLastFrameTracks())
            {
                const auto &box = track.box_history.back();
                outFile << track.id << " " << box.xmin << " " << box.ymin << " " << box.xmax << " " << box.ymax << std::endl;
                if (show || save)
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
        if (show)
        {
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(box->w(), box->h()));
            Fl_RGB_Image *newImage = CopyMat2FlImage(resized);
            delete (Fl_RGB_Image *)box->image();
            box->image(newImage);
            box->redraw();
            Fl::check();
        }
        if (save)
        {
            cv::imwrite(image_save_path + "/" + std::to_string(frameNumber) + ".jpg", frame);
        }
        frameNumber++;
    }
    cap.release();
    Fl_RGB_Image *tmp = CopyMat2FlImage(bg_img);
    delete (Fl_RGB_Image *)box->image();
    box->image(tmp);
    box->redraw();
    Fl::check();
}

int main(int argc, char **argv)
{
    generateColors(100);

    Fl_Window *window = new Fl_Window(1800, 1000, "Track Athletes");

    Fl_Button *button = new Fl_Button(80, 20, 200, 50, "Choose video");

    Fl_Output* output = new Fl_Output(80, 80, 200, 25);  // New output widget to display the selected file path

    Fl_Choice* model_choice = new Fl_Choice(80, 140, 200, 25, "Model:");
    model_choice->add("football model");
    model_choice->add("tennis model");
    model_choice->value(1);
    model_choice->callback(model_choice_cb);
    model_option = "tennis";
    

    Fl_Choice* inference_choice = new Fl_Choice(80, 200, 200, 25, "Inference:");
    inference_choice->add("detection only");
    inference_choice->add("tracking");
    inference_choice->value(1);  // Set the default choice to "tracking"
    inference_choice->callback(inference_choice_cb);
    inference_option = "tracking";

    Fl_Check_Button *check = new Fl_Check_Button(80, 240, 300, 50, "Show while processing");
    check->value(1);  // Set the initial state to ticked
    check->callback(check_cb, nullptr);

    Fl_Button *start_button = new Fl_Button(80, 300, 200, 50, "Start");  // New start button

    Fl_Box *box = new Fl_Box(400, 20, 1360, 960);
    cv::Mat img = cv::imread("../data/example.jpg");
    cv::resize(img, img, cv::Size(1360, 940));
    Fl_RGB_Image *tmp = CopyMat2FlImage(img);
    box->image(tmp);

    VideoProcessor processor(box);
    button->callback(button_cb, output);
    start_button->callback(start_cb, &processor); 

    window->end();
    window->show(argc, argv);

    return Fl::run();
}
