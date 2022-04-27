#include "zv/Client.h"

#include "Utils.h"
#include "dl_opencv.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void logImage(const std::string& name, const cv::Mat3b& im)
{
    static cv::Mat4b tmpRgba;
    cv::cvtColor(im, tmpRgba, cv::COLOR_BGR2RGBA);
    zv::logImageRGBA (name, tmpRgba.data, tmpRgba.cols, tmpRgba.rows, tmpRgba.step);
}

void logImage(const std::string& name, const cv::Mat1b& im)
{
    static cv::Mat4b tmpRgba;
    cv::cvtColor(im, tmpRgba, cv::COLOR_GRAY2RGBA);
    zv::logImageRGBA (name, tmpRgba.data, tmpRgba.cols, tmpRgba.rows, tmpRgba.step);
}

int main (int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " image_prefix" << std::endl;
        std::cerr << "It will load image_prefix.aliased.png and image_prefix.antialiased.png to generate image_prefix.json" << std::endl;
        return 1;
    }

    const bool debug = false;
    if (debug && !zv::launchServer ())
    {
        std::cerr << "Could not launch zv" << std::endl;
        return 2;
    }
    
    std::string fileprefix = argv[1];

    cv::Mat3b antialiased = cv::imread (fileprefix + ".antialiased.png", cv::IMREAD_COLOR);
    cv::Mat3b aliased = cv::imread (fileprefix + ".aliased.png", cv::IMREAD_COLOR);

    const int rows = aliased.rows;
    const int cols = aliased.cols;
    assert (antialiased.rows == rows);
    assert (antialiased.cols == cols);

    logImage ("antialiased", antialiased);
    logImage ("aliased", aliased);

    cv::Mat1b mask_changed (rows, cols);
    for_all_rc (aliased)
    {
        cv::Vec3i d3i = cv::Vec3i(aliased(r,c)) - cv::Vec3i(antialiased(r,c));
        bool changed = (d3i[0] != 0 || d3i[1] != 0 || d3i[2] != 0);
        mask_changed(r,c) = changed ? 255 : 0;
    }

    logImage ("mask_changed", mask_changed);
    
    std::vector<int> ordered_labels;
    ordered_labels.reserve(255);
    // First add multiples of 16 so we can see something.
    for (int i = 16; i < 256; i += 16)
    {
        ordered_labels.push_back(i);
    }
    // If that's not enough, keep filling with the intermediate values.
    for (int i = 1; i < 255; ++i)
    {
        if (i % 16 != 0)
            ordered_labels.push_back(i);
    }

    auto next_label_it = ordered_labels.begin();
    std::unordered_map<cv::Vec3b, int, cv::vec3b_hash> label_map;
    
    
    std::vector<std::pair<int, cv::Vec3b>> finalLabels;
    // Add the background.
    finalLabels.push_back (std::make_pair(0, cv::Vec3b(255,255,255)));
    
    cv::Mat1b label_image (rows, cols);
    for_all_rc (mask_changed)
    {
        int label = 255; // means unassigned. 0 means background.

        if (mask_changed(r,c))
        {
            cv::Vec3b color = aliased(r,c);
            auto label_it = label_map.find (color);
            if (label_it == label_map.end())
            {
                label = *next_label_it;
                label_map.insert (std::make_pair (color, *next_label_it));
                finalLabels.push_back (std::make_pair(label, color));
                ++next_label_it;
                if (next_label_it == ordered_labels.end())
                {
                    std::cerr << "Too many labels, aborting." << std::endl;
                    return 1;
                }
            }
            else
            {
                label = label_it->second;
            }
        }

        label_image(r,c) = label;
    }

    logImage ("labels_before_fill", label_image);

    const bool fillFgMask = true;
    if (fillFgMask)
    {
        std::deque<cv::Point> fgPoints;

        auto enqueue_neighbs = [&](int r, int c) {
            for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc)
            {
                if (dr == 0 && dc == 0)
                    continue;
                int nr = r + dr;
                int nc = c + dc;
                if (nr < 0 || nr >= label_image.rows || nc < 0 || nc >= label_image.cols)
                    continue;
                if (label_image(nr,nc) == 255)
                    fgPoints.push_back (cv::Point(nc,nr));
            }
        };

        for_all_rc (label_image)
        {
            if (label_image(r,c) != 255)
            {
                enqueue_neighbs (r,c);
            }
        }

        while (!fgPoints.empty())
        {
            cv::Point p = fgPoints.front();
            fgPoints.pop_front();
            auto label_it = label_map.find (aliased(p.y,p.x));
            if (label_it == label_map.end())
            {
                label_image(p.y,p.x) = 0; // background.
                continue;
            }

            label_image(p.y,p.x) = label_it->second;
            enqueue_neighbs(p.y, p.x);
        }
    }

    // All remaining unknown pixels are background
    for_all_rc (label_image)
    {
        if (label_image(r,c) == 255)
        {
            label_image(r,c) = 0;
        }
    }

    logImage ("labels_after_fill", label_image);

    std::string jsonFile;
    jsonFile += "{";
    jsonFile += dl::formatted("\"size_cols_rows\":[%d, %d],", cols, rows); 
    jsonFile += dl::formatted("\"labels\":[");
    for (int i = 0; i < finalLabels.size(); ++i)
    {
        const auto& labelAndColor = finalLabels[i];
        jsonFile += dl::formatted("{\"label\":%d,\"rgb_color\":[%d,%d,%d]}",
                                  labelAndColor.first, labelAndColor.second[0], labelAndColor.second[1], labelAndColor.second[2]);
        if (i < finalLabels.size()-1)
            jsonFile += ',';
    }
    jsonFile += dl::formatted("]");
    jsonFile += "}\n";
    
    {
        std::ofstream f (fileprefix + ".json");
        f << jsonFile;
    }

    cv::imwrite(fileprefix + ".labels.png", label_image);

    zv::waitUntilDisconnected ();
    return 0;
}
