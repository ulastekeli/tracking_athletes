cmake_minimum_required(VERSION 3.10)
project(Tracker)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenCV REQUIRED)
# Find the FLTK and FLTK_IMAGES libraries
# find_library(FLTK_LIB
#              NAMES fltk
#              HINTS ${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)
# find_library(FLTK_IMAGES_LIB
#              NAMES fltk_images
#              HINTS ${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)




# include_directories(${FLTK_INCLUDE_DIRS})
# message(STATUS "FLTK_INCLUDE_DIRS: ${FLTK_INCLUDE_DIRS}")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/opencv/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/fltk/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dmlc/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dlpack/include)

link_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/libs)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/opencv/lib)

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/Box.cpp
    ${PROJECT_SOURCE_DIR}/src/Detector.cpp
    ${PROJECT_SOURCE_DIR}/src/Reid.cpp
    ${PROJECT_SOURCE_DIR}/src/Track.cpp
    ${PROJECT_SOURCE_DIR}/src/Tracker.cpp
)

add_library(tracking ${SOURCES})
target_link_libraries(tracking opencv_stitching opencv_aruco opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_dnn opencv_dnn_objdetect opencv_dnn_superres opencv_dpm opencv_highgui opencv_face opencv_freetype opencv_fuzzy opencv_hfs opencv_img_hash opencv_line_descriptor opencv_quality opencv_reg opencv_rgbd opencv_saliency opencv_shape opencv_stereo opencv_structured_light opencv_phase_unwrapping opencv_superres opencv_optflow opencv_surface_matching opencv_tracking opencv_datasets opencv_text opencv_dnn opencv_plot opencv_ml opencv_videostab opencv_videoio opencv_ximgproc opencv_video opencv_xobjdetect opencv_objdetect opencv_calib3d opencv_imgcodecs opencv_features2d opencv_flann opencv_xphoto opencv_photo opencv_imgproc opencv_core fltk fltk_images tvm_runtime dl)
# target_link_libraries(libs )

add_executable(athleteTrackingApp ${PROJECT_SOURCE_DIR}/app.cpp)
target_link_libraries(athleteTrackingApp tracking)
target_include_directories(athleteTrackingApp PRIVATE "/usr/include")

add_executable(opencv_detector_test ${PROJECT_SOURCE_DIR}/tests/opencv_detector_test.cpp)
target_link_libraries(opencv_detector_test tracking)
target_include_directories(opencv_detector_test PRIVATE "/usr/include")

add_executable(app_test ${PROJECT_SOURCE_DIR}/tests/app_test.cpp)
target_link_libraries(app_test tracking)
target_include_directories(app_test PRIVATE "/usr/include")

add_executable(reid_test ${PROJECT_SOURCE_DIR}/tests/reid_test.cpp)
target_link_libraries(reid_test tracking)
target_include_directories(reid_test PRIVATE "/usr/include")



