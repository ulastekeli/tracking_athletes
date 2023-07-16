cmake_minimum_required(VERSION 3.10)
project(Tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
# Find the FLTK and FLTK_IMAGES libraries
find_library(FLTK_LIB
             NAMES fltk
             HINTS "/usr/lib/x86_64-linux-gnu")
find_library(FLTK_IMAGES_LIB
             NAMES fltk_images
             HINTS "/usr/lib/x86_64-linux-gnu")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FLTK_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dmlc/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dlpack/include)

link_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/libs)

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/Box.cpp
    ${PROJECT_SOURCE_DIR}/src/Detector.cpp
    ${PROJECT_SOURCE_DIR}/src/Reid.cpp
    ${PROJECT_SOURCE_DIR}/src/Track.cpp
    ${PROJECT_SOURCE_DIR}/src/Tracker.cpp
)

add_library(libs ${SOURCES})
target_link_libraries(libs ${OpenCV_LIBS} ${FLTK_LIB} ${FLTK_IMAGES_LIB} tvm_runtime dl)
# target_link_libraries(libs )

add_executable(opencv_detector_test ${PROJECT_SOURCE_DIR}/tests/opencv_detector_test.cpp)
add_executable(app_test ${PROJECT_SOURCE_DIR}/tests/app_test.cpp)
add_executable(reid_test ${PROJECT_SOURCE_DIR}/tests/reid_test.cpp)
add_executable(app ${PROJECT_SOURCE_DIR}/app.cpp)

target_link_libraries(app libs)
target_link_libraries(app_test libs)
target_link_libraries(opencv_detector_test libs)
target_link_libraries(reid_test libs)

target_include_directories(app PRIVATE "/usr/include")
target_include_directories(app_test PRIVATE "/usr/include")
target_include_directories(opencv_detector_test PRIVATE "/usr/include")
target_include_directories(reid_test PRIVATE "/usr/include")

