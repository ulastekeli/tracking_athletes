cmake_minimum_required(VERSION 3.10)
project(Tracker)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
# Find the FLTK and FLTK_IMAGES libraries
find_library(FLTK_LIB
             NAMES fltk
             HINTS "/usr/lib/x86_64-linux-gnu")
find_library(FLTK_IMAGES_LIB
             NAMES fltk_images
             HINTS "/usr/lib/x86_64-linux-gnu")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${FLTK_INCLUDE_DIRS})

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/Box.cpp
    ${PROJECT_SOURCE_DIR}/src/Detector.cpp
    ${PROJECT_SOURCE_DIR}/src/Track.cpp
)

add_library(libs ${SOURCES})
target_link_libraries(libs ${OpenCV_LIBS} ${FLTK_LIB} ${FLTK_IMAGES_LIB})
# target_link_libraries(libs )

add_executable(test ${PROJECT_SOURCE_DIR}/test.cpp)
add_executable(old ${PROJECT_SOURCE_DIR}/old_detector.cpp)
add_executable(app app.cpp)

# target_link_libraries(app libs)
target_link_libraries(app libs)
target_link_libraries(test libs)
target_link_libraries(old libs)
target_include_directories(app PRIVATE "/usr/include")

