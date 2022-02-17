#pragma once

#include <opencv2/opencv.hpp>

#define for_all_rc(im) \
    for (int r = 0; r < im.rows; ++r) \
    for (int c = 0; c < im.cols; ++c)

namespace cv
{

inline bool operator< (const cv::Vec3b& lhs, const cv::Vec3b& rhs)
{
    for (int k = 0; k < 3; ++k)
    {
        if (lhs[k] < rhs[k])
            return true;
        
        if (rhs[k] > lhs[k])
            return false;
    }
    
    // equality
    return false;
}

} // cv

