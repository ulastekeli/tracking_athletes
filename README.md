# Athlete Tracking System

This project is a real-time video processing application that uses machine learning to track athletes in sports videos. The application is built on the Fast Light Toolkit (FLTK) and OpenCV, and includes support for object detection and tracking. It provides an easy-to-use graphical interface where users can choose video files and configure the system to either perform detection only or tracking.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

The application is built using CMake and requires OpenCV, FLTK, and a C++ compiler supporting the C++17 standard.

To build and run the project, follow these steps:

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
# This step is required if opencv is not installed
sudo cp 3rd_party/opencv/lib/* /usr/local/lib
# Create a build directory and navigate into it:
mkdir build
cd build
# Configure the project with CMake:
cmake ..
# Build the project:
make
# Run the application:
./athleteTrackingApp
```

Please note that you need to replace <repository_url> and <repository_directory> with the URL of this repository and the name of the directory where you want to clone the repository, respectively.

## Usage

The application provides a simple user interface to choose a video file and select the mode of operation (detection only or tracking).

1. Click on the "Choose video" button and select the video file you want to process. The file path will be displayed in the application window.

2. Select the model you want to use from the "Model" dropdown menu. You can choose between "football model" and "tennis model".

3. Select the mode of operation from the "Inference" dropdown menu. You can choose between "detection only" and "tracking".

4. If you want to view the processing results in real-time, check the "Show while processing" box.

5. Click the "Start" button to start the video processing.

The application will then process the video file and display the results either in real-time or save them in the specified output directory depending on your settings.
