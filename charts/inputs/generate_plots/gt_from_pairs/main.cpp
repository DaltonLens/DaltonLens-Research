#include "zv/Client.h"

#include "Utils.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void logImage(const std::string& name, const cv::Mat3b& im)
{
    static cv::Mat4b tmpRgba;
    cv::cvtColor(im, tmpRgba, cv::COLOR_BGR2RGBA);
    dl_dbg ("%d x %d, step = %d", (int)tmpRgba.cols, (int)tmpRgba.rows, (int)tmpRgba.step);
    zv::logImageRGBA (name, tmpRgba.data, tmpRgba.cols, tmpRgba.rows, tmpRgba.step);
}

int main (int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " antialiased_image aliased_image" << std::endl;
        return 1;
    }

    if (!zv::launchServer ())
    {
        std::cerr << "Could not launch zv" << std::endl;
        return 2;
    }
    
    cv::Mat3b antialiased = cv::imread (argv[1], cv::IMREAD_COLOR);
    cv::Mat3b aliased = cv::imread (argv[2], cv::IMREAD_COLOR);

    logImage ("antialiased", antialiased);
    logImage ("aliased", aliased);

    zv::waitUntilDisconnected ();
    return 0;
}
