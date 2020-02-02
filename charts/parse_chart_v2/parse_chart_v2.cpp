/*

New idea: color quantization first. E.g. median cut algorithm. Then OpenGM graphcut with alpha-expansion
Threshold pixels with small gradient. Closing to keep large regions.
Histogram of colors for low-gradient areas. Largest region is background color.
Compute distance from background color. Remove points with significantly lower distance than some of their 3x3 neighbors
Color quantization from those maxima. Keep top 16 or 32 colors.
Build a graph. Neighbor pixels connected. Cost of being different depends on their alpha-distance. Cost to the labels depend on the alpha-distance. Weight is so that a higher distance from background gives it a higher weight, and prefer the closest color with stronger alpha. Idea is to strongly attach strong pixels to the true label, and have the other ones connect to them via regularization.
Optimize via alpha-expansion.


*/

#include <opencv2/opencv.hpp>

#include "dl_opencv.h"
#include "dl_quantization.h"

#include <unordered_map>
#include <utility>
#include <cstdio>

using namespace cv;

inline short rgbDist(cv::Vec3b lhs, cv::Vec3b rhs)
{
    return std::abs(rhs[0]-lhs[0]) + std::abs(rhs[1]-lhs[1]) + std::abs(rhs[2]-lhs[2]);
}

int alphaAwareDistWithKnownAlpha(const cv::Vec3b currRgb, 
                                 const cv::Vec3b otherRgb, 
                                 const cv::Vec3b currBgColor,
                                 const float alpha1,
                                 const float alpha2)
{
    Vec3b rgb1;
    Vec3b rgb2;
    for (int k = 0; k < 3; ++k)
    {
        float v1 = currBgColor[k] + (currRgb[k] - currBgColor[k]) / alpha1;
        v1 = std::max(std::min(v1, 255.f), 0.f);
        rgb1[k] = int(v1 + 0.5f);

        float v2 = currBgColor[k] + (otherRgb[k] - currBgColor[k]) / alpha2;
        v2 = std::max(std::min(v2, 255.f), 0.f);
        rgb2[k] = int(v2 + 0.5f);
    }

    return rgbDist(rgb1, rgb2);
}

std::tuple<int, float, float>
alphaAwareDist(const cv::Vec3b currRgb, const cv::Vec3b otherRgb, const cv::Vec3b currBgColor)
{
    int minDist = std::numeric_limits<int>::max();
    float bestCoarseAlpha1 = NAN;
    float bestCoarseAlpha2 = NAN;

    const int numCoarseSteps = 16;
    const float coarseStep = 1.0f/numCoarseSteps;

    // We want to go from step to 1.0.
    // Zero has no relevant here.
    for (int i = 1; i <= numCoarseSteps; ++i)
    for (int j = 1; j <= numCoarseSteps; ++j)
    {
        const float alpha1 = i*coarseStep;
        const float alpha2 = j*coarseStep;
        
        const int d = alphaAwareDistWithKnownAlpha(currRgb, otherRgb, currBgColor, alpha1, alpha2);
        // fprintf (stderr, "%f %f -> %d\n", alpha1, alpha2, d);
        if (d < minDist)
        {
            minDist = d;
            bestCoarseAlpha1 = alpha1;
            bestCoarseAlpha2 = alpha2;
        }
    }

    const int numFineStepsDiv2 = 8;
    const float fineStep = float(coarseStep)/(numFineStepsDiv2*2);

    float bestFineAlpha1 = bestCoarseAlpha1;
    float bestFineAlpha2 = bestCoarseAlpha2;
    for (int i = -numFineStepsDiv2; i <= numFineStepsDiv2; ++i)
    for (int j = -numFineStepsDiv2; j <= numFineStepsDiv2; ++j)
    {
        const float alpha1 = bestCoarseAlpha1 + i*fineStep;
        const float alpha2 = bestCoarseAlpha2 + j*fineStep;

        const int d = alphaAwareDistWithKnownAlpha(currRgb, otherRgb, currBgColor, alpha1, alpha2);
        // fprintf (stderr, "%f %f -> %d\n", alpha1, alpha2, d);
        if (d < minDist)
        {
            minDist = d;
            bestFineAlpha1 = alpha1;
            bestFineAlpha2 = alpha2;
        }
    }

    return { minDist, bestFineAlpha1, bestFineAlpha2 };
}

namespace std
{
    template <>
    struct hash<cv::Vec3b>
    {
        uint64_t operator()(cv::Vec3b rgb) const
        {
            return std::hash<int32_t>()(rgb[0] + (rgb[1]<<8) + (rgb[2]<<16));
        }
    };
    
} // std

using MouseAction = std::function<void(int, int, int, int)>;

void setMouseAction(const std::string& winName, const MouseAction& action)
{
    cv::setMouseCallback(winName,
                         [] (int event, int x, int y, int flags, void* userdata) {
                             (*(MouseAction*)userdata)(event, x, y, flags);
                         }, (void*)&action);
}

int main (int argc, char* argv[])
{
    if (false)
    {
        cv::Vec3b bgColor (107, 27, 135);
        
        cv::Vec3b lightFg (101, 56, 135);
        cv::Vec3b strongFg (76, 177, 133);
        cv::Vec3b trueFg (61, 245, 132);
        
        cv::Vec3b lightOther (104, 69, 162);
        cv::Vec3b strongOther (98, 154, 216);
        
        int dist; float alpha1, alpha2;
        std::tie(dist, alpha1, alpha2) = alphaAwareDist(trueFg,
                                                        strongFg,
                                                        bgColor);
        
        fprintf (stderr, "alphaDist = %d alpha1=%f alpha2=%f\n",
                 dist, alpha1, alpha2);
        
        return 0;
    }
    
    if (argc != 2)
    {
        std::cerr << "Usage: parse_chart_v2 input_image" << std::endl;
        return 1;
    }

    cv::Mat3b im = imread(argv[1]);
    assert (im.data);

    cv::imshow("im", im);
    
    {
        static MouseAction im_onMouse = [im]( int event, int x, int y, int ) {
            if (event != EVENT_LBUTTONDOWN)
                return;
            
            auto bgr = im(y,x);
            fprintf (stderr, "[%d %d] -> [%d %d %d]", x, y, bgr[2], bgr[1], bgr[0]);
            
            cv::Vec3b bgColor (135, 27, 107);
            
            for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc)
            {
                auto neighb = im(y+dr, x+dc);
                
                int dist; float alpha1, alpha2;
                std::tie(dist, alpha1, alpha2) = alphaAwareDist(bgr,
                                                                neighb,
                                                                bgColor);
                
                fprintf (stderr, "\t[%d %d] -> %d (%.2f %.2f)\n",
                         dr, dc,
                         dist, alpha1, alpha2);
            }
        };
        setMouseAction( "im", im_onMouse );
        cv::waitKey(0);
    }

    cv::Mat3s gradientMagnitude;
    cv::Sobel(im, gradientMagnitude, CV_16S, 1, 1);

    // Low gradient pixels.
    cv::Mat1b lowGradPixels (im.rows, im.cols);
    lowGradPixels = 0;
    for_all_rc (gradientMagnitude)
    {
        const float magn = std::abs(gradientMagnitude(r,c)[0]) + std::abs(gradientMagnitude(r,c)[1]) + std::abs(gradientMagnitude(r,c)[2]);
        if (magn < 20)
            lowGradPixels(r,c) = 255;
    }

    // Make sure we're safe.
    cv::Mat1b erodedLowGradPixels;
    cv::erode (lowGradPixels, erodedLowGradPixels, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));

    cv::imshow("lowGradPixels", lowGradPixels);
    cv::imshow("erodedLowGradPixels", erodedLowGradPixels);

    cv::waitKey(1);

    // FIXME: assuming a perfect and constant background for now.
    // We should make the bins larger and then take the mean of the
    // pixels in the bin to be a bit better. But to handle gradients
    // we'll need to do a connected component region growing.
    std::unordered_map<cv::Vec3b, std::vector<int>> lowGradHistogram;
    for_all_rc (erodedLowGradPixels)
    {
        if (erodedLowGradPixels(r,c))
        {
            auto rgb = im(r,c);
            // Divide the colors by 4 to make the bins larger.
            lowGradHistogram[rgb].push_back(r*im.cols+c);
        }
    }

    auto compareVectorSize = [](const auto& lhs, const auto& rhs) { return lhs.second.size() < rhs.second.size(); };
    const auto max_it = std::max_element(lowGradHistogram.begin(), lowGradHistogram.end(), compareVectorSize);
    
    // Assuming a constant bg color for now
    cv::Vec3b bgColor = max_it->first;
    cv::Mat3b bgColorIm (im.cols, im.rows);
    bgColorIm = bgColor;
        
    cv::imshow("bgColorIm", bgColorIm);

    cv::Mat1b bgMask (im.cols, im.rows);
    bgMask = 0;
    for_all_rc (im)
    {
        if (rgbDist(bgColorIm(r,c), im(r,c)) < 10)
          bgMask(r,c) = 255;  
    }

    cv::imshow("bgMask", bgMask);

    cv::Mat1f alphaImage (im.cols, im.rows);
    alphaImage = NAN;

    // Assume that pixels with low-grad have their true color.
    for_all_rc (im)
    {
        if (lowGradPixels(r,c))
            alphaImage(r,c) = 1.0;
    }

    MedianCut quantizer;
    std::vector<cv::Vec3b> palette;
    cv::Mat1b indexed;
    const int numColors = 64;
    quantizer.apply (im, numColors, indexed, palette);
    cv::Mat3b indexedAsRgb = indexedToRgb(indexed, palette);

    fprintf (stderr, "Palette: \n");
    for (const auto& rgb : palette)
        fprintf (stderr, "\t[%d %d %d]\n", rgb[0], rgb[1], rgb[2]);
    
    cv::imshow("indexed", indexed);
    cv::imshow("quantized", indexedAsRgb);
    cv::waitKey(0);

    return 0;
}
