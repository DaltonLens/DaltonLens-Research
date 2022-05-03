#include "zv/Client.h"

#include "Utils.h"
#include "dl_opencv.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>

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

struct PerColorInfo
{
    int label = -1;
    int count = 0;
};

int main (int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " antialiased_image aliased_image output_prefix" << std::endl;
        return 1;
    }

    const bool debug = false;
    if (debug && !zv::launchServer ())
    {
        std::cerr << "Could not launch zv" << std::endl;
        return 2;
    }
    
    cv::Mat3b antialiased = cv::imread (argv[1], cv::IMREAD_COLOR);
    cv::Mat3b aliased = cv::imread (argv[2], cv::IMREAD_COLOR);

    const std::string outfileprefix = argv[3];

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
    cv::Mat1b mask_background = 255 - mask_changed;
    
    // cv::Mat1f dist_from_change_1f;
    // cv::distanceTransform(mask_background, dist_from_change_1f, cv::DIST_L2, 3);
    // cv::Mat1b dist_from_change (rows, cols);
    // for_all_rc (dist_from_change)
    // {
    //     dist_from_change(r,c) = int(std::min(dist_from_change_1f(r,c), 255.0f) + 0.5f);
    // }
    
    // logImage ("dist_from_change", dist_from_change);

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
    std::unordered_map<cv::Vec3b, PerColorInfo, cv::vec3b_hash> label_map;
    
    cv::Mat1b label_image (rows, cols);
    label_image = 255; // unknown.

    cv::Vec3b uniform_background = cv::Vec3b(255,255,255);
    bool found_uniform_background = false;
    {
        std::unordered_map<cv::Vec3b, int, cv::vec3b_hash> numPixelsPerBorderColor;
        int numBorderPixels = 0;
        for_all_rc (aliased)
        {
            // Only process the border.
            if (r > rows*0.1 && r < rows*0.9 && c > cols*0.1 && c < cols*0.9)
                continue;
            numPixelsPerBorderColor[aliased(r,c)] += 1;
            ++numBorderPixels;
        }


        auto max_color_it = std::max_element(numPixelsPerBorderColor.begin(), numPixelsPerBorderColor.end(),
                                             [](const std::pair<cv::Vec3b, int> &a,
                                                const std::pair<cv::Vec3b, int> &b)
                                             {
                                                 return a.second < b.second;
                                             });

        if (max_color_it->second > numBorderPixels * 0.5)
        {
            uniform_background = max_color_it->first;
            found_uniform_background = true;
            fprintf (stdout, "Found a uniform background in the border area. (%d %d %d)\n",
                    uniform_background[0], uniform_background[1], uniform_background[2]);
            label_map[uniform_background].label = 0;
            for_all_rc (label_image)
            {
                if (aliased(r,c) == uniform_background)
                {
                    label_image(r,c) = 0;
                    label_map[uniform_background].count += 1;
                }
            }
        }
    }

    std::vector<std::pair<int, cv::Vec3b>> finalLabels;
    // Add the background.
    finalLabels.push_back (std::make_pair(0, uniform_background));
    
    for_all_rc (mask_changed)
    {
        // Already assigned to background.
        if (label_image(r,c) != 255)
            continue;

        int label = 255; // means unassigned. 0 means background.

        if (mask_changed(r,c))
        {
            cv::Vec3b color = aliased(r,c);
            auto label_it = label_map.find (color);
            if (label_it == label_map.end())
            {
                label = *next_label_it;
                PerColorInfo colorInfo;
                colorInfo.label = *next_label_it;
                colorInfo.count = 1;
                label_map.insert (std::make_pair (color, colorInfo));
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
                label = label_it->second.label;
                label_it->second.count += 1;
            }
        }

        label_image(r,c) = label;
    }

    logImage ("labels_before_fill", label_image);

    // Fill pixels that did not change
    {
        for_all_rc (label_image)
        {
            if (label_image(r,c) != 255)
                continue;
                        
            auto label_it = label_map.find (aliased(r,c));
            if (label_it != label_map.end())
            {
                label_image(r,c) = label_it->second.label;
                ++label_it->second.count;
            }
        }
    }

    // All remaining unknown pixels are background
    int num_background = 0;
    for_all_rc (label_image)
    {
        if (label_image(r,c) == 255)
        {
            label_image(r,c) = 0;
            if (!found_uniform_background)
                uniform_background = aliased(r,c);
        }

        if (label_image(r,c) == 0)
            ++num_background;
    }

    if (num_background == 0)
    {
        auto max_color_it = std::max_element(label_map.begin(), label_map.end(),
                                             [](const std::pair<cv::Vec3b, PerColorInfo> &a,
                                                const std::pair<cv::Vec3b, PerColorInfo> &b)
                                             {
                                                 return a.second.count < b.second.count;
                                             });

        int labelToReplace = max_color_it->second.label;
        dl_dbg ("Replacing label %d with background", labelToReplace);
        for_all_rc (label_image)
        {
            if (label_image(r,c) == labelToReplace)
                label_image(r,c) = 0;
            label_map.erase (labelToReplace);
            std::remove_if(finalLabels.begin(), finalLabels.end(), [labelToReplace](const std::pair<int, cv::Vec3b>& labelAndColor) {
                return (labelAndColor.first == labelToReplace);
            });
        }

        finalLabels[0].second = max_color_it->first;
    }
    else
    {
        finalLabels[0].second = uniform_background;
    }

    logImage ("labels_after_fill", label_image);


    // Disabled this as it leads to more confusions. Large uniform areas
    // will still needs to be reconstructed properly. Hopefully they'll be
    // rare enough to avoid dominating the error terms.
    // // Make sure that pixels that are far from a change are all
    // // marked as background.
    // for_all_rc (label_image)
    // {
    //     if (dist_from_change(r,c) > 2)
    //         label_image (r,c) = 0;
    // }
    // logImage ("labels_after_dist", label_image);
    
    fprintf (stdout, "Number of labels: %d\n", (int)finalLabels.size());

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
        std::ofstream f (outfileprefix + ".json");
        f << jsonFile;
    }

    cv::imwrite(outfileprefix + ".labels.png", label_image);

    zv::waitUntilDisconnected ();
    return 0;
}
