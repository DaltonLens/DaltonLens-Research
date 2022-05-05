cmake_minimum_required(VERSION 3.0)
project(gt_from_pairs)
set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS})

add_compile_options(
    $<$<CONFIG:Debug>:-DDEBUG>
)

add_executable(gt_from_pairs 
    dl_opencv.h
    Utils.h
    Utils.cpp
    main.cpp

    zv/Client.cpp
    zv/znet.cpp
    zv/Message.h
    zv/Client.h
    zv/znet.h
    zv/znet_zv.h
)

target_compile_definitions(gt_from_pairs PRIVATE -DGL_SILENCE_DEPRECATION=1)
target_link_libraries(gt_from_pairs ${OpenCV_LIBS} pthread)
