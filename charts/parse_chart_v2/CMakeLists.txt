cmake_minimum_required(VERSION 3.0)
project(parse_chart_v2)
set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS})

find_package( GLEW REQUIRED )
find_package( glfw3 REQUIRED )
set (IMGUI_CVLOG_DIR ../../../imgui-cvlog)
include_directories ("${IMGUI_CVLOG_DIR}" "${IMGUI_CVLOG_DIR}/imgui" "${IMGUI_CVLOG_DIR}/implot")

set (opengm_DIR /opt/opengm/lib/cmake/opengm)
find_package( opengm REQUIRED )

include_directories (/usr/local/include ${OPENGM_INCLUDE_DIRS})

add_compile_options(
            $<$<CONFIG:Debug>:-O3>
        )

set (imgui_cvlog_sources 
    ${IMGUI_CVLOG_DIR}/imgui_cvlog.h
    ${IMGUI_CVLOG_DIR}/imgui_cvlog.cpp
    ${IMGUI_CVLOG_DIR}/glfw_opencv/imgui_cvlog_gl_opencv.h
    ${IMGUI_CVLOG_DIR}/glfw_opencv/imgui_cvlog_gl_opencv.cpp

    ${IMGUI_CVLOG_DIR}/imgui/imconfig.h
    ${IMGUI_CVLOG_DIR}/imgui/imgui.h
    ${IMGUI_CVLOG_DIR}/imgui/imgui.cpp
    ${IMGUI_CVLOG_DIR}/imgui/imgui_draw.cpp
    ${IMGUI_CVLOG_DIR}/imgui/imgui_widgets.cpp

    ${IMGUI_CVLOG_DIR}/implot/implot.h
    ${IMGUI_CVLOG_DIR}/implot/implot.cpp
    ${IMGUI_CVLOG_DIR}/implot/implot_items.cpp
    ${IMGUI_CVLOG_DIR}/imgui/examples/imgui_impl_opengl3.cpp
    ${IMGUI_CVLOG_DIR}/imgui/examples/imgui_impl_glfw.cpp
)

add_executable(parse_chart_v2 
    dl_opencv.h
    dl_quantization.h
    Utils.h
    Utils.cpp

    parse_chart_v2.cpp

    ${imgui_cvlog_sources}
)

target_compile_definitions(parse_chart_v2 PRIVATE -DGL_SILENCE_DEPRECATION=1)
target_link_libraries(parse_chart_v2 ${OpenCV_LIBS} glfw GLEW::glew)
