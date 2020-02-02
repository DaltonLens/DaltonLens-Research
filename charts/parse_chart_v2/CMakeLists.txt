cmake_minimum_required(VERSION 3.0)
project(parse_chart_v2)
set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS})

add_executable(parse_chart_v2 
    dl_opencv.h
    dl_quantization.h
    parse_chart_v2.cpp
)

target_link_libraries(parse_chart_v2 ${OpenCV_LIBS} )
