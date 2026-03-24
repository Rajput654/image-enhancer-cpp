cmake_minimum_required(VERSION 3.10)
project(ImageEnhancer CXX)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for speed in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

# Find OpenCV (Requires core, imgproc, imgcodecs, objdetect, dnn)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Create the executable
add_executable(enhance_app main.cpp)

# Link OpenCV libraries
target_link_libraries(enhance_app ${OpenCV_LIBS})