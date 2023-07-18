cmake_minimum_required(VERSION 3.10)
project(Tracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
# Find the FLTK and FLTK_IMAGES libraries
# find_library(FLTK_LIB
#              NAMES fltk
#              HINTS ${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)
# find_library(FLTK_IMAGES_LIB
#              NAMES fltk_images
#              HINTS ${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${FLTK_INCLUDE_DIRS})
# message(STATUS "FLTK_INCLUDE_DIRS: ${FLTK_INCLUDE_DIRS}")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/fltk/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dmlc/include)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/dlpack/include)

link_directories(${PROJECT_SOURCE_DIR}/3rd_party/tvm/libs)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/fltk/lib)

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/Box.cpp
    ${PROJECT_SOURCE_DIR}/src/Detector.cpp
    ${PROJECT_SOURCE_DIR}/src/Reid.cpp
    ${PROJECT_SOURCE_DIR}/src/Track.cpp
    ${PROJECT_SOURCE_DIR}/src/Tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/TvmDetector.cpp
)

add_library(tracking ${SOURCES})
target_link_libraries(tracking ${OpenCV_LIBS} fltk fltk_images tvm_runtime dl)
# target_link_libraries(libs )

add_executable(app ${PROJECT_SOURCE_DIR}/app.cpp)
target_link_libraries(app tracking)
target_include_directories(app PRIVATE "/usr/include")

add_executable(opencv_detector_test ${PROJECT_SOURCE_DIR}/tests/opencv_detector_test.cpp)
target_link_libraries(opencv_detector_test tracking)
target_include_directories(opencv_detector_test PRIVATE "/usr/include")

add_executable(app_test ${PROJECT_SOURCE_DIR}/tests/app_test.cpp)
target_link_libraries(app_test tracking)
target_include_directories(app_test PRIVATE "/usr/include")

add_executable(reid_test ${PROJECT_SOURCE_DIR}/tests/reid_test.cpp)
target_link_libraries(reid_test tracking)
target_include_directories(reid_test PRIVATE "/usr/include")

add_executable(tvm_detector_test ${PROJECT_SOURCE_DIR}/tests/tvm_detector_test.cpp)
target_link_libraries(tvm_detector_test tracking)
target_include_directories(tvm_detector_test PRIVATE "/usr/include")


