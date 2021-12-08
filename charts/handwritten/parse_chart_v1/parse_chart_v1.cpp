/*

Implement a first simple method -> parse_chart_v1, abandoned
1) Segment the background
2) [Optional, later] Determine the background for each pixel, interpolating to handle gradients
3) Fix alpha to 1.0 for pixels that have a low 5x5 gradient (color splats, large line widths)
4) For each pixel, determine to which neighbors it's connected. Minimize the error with N alphas and one RGB in a 3x3 neighborhood. Compute the alpha distance to each neighbor and threshold it. Store the connectivity in a bitmask.
5) Resolve the connectivity for neighbor pixels that both consider each other as a friend.
6) Now do region growing to extract connected components, seeding from foreground pixels.
7) Determine the best color for each region, solving for alpha and rgb
8) Merge regions if their alpha-aware distance is similar. This will wrongly include intensity-based differences, but color-blind people are good as dicriminating those..
9) Output the mask

*/

#include <opencv2/opencv.hpp>

#include <unordered_map>
#include <utility>
#include <cstdio>

using namespace cv;

#define for_all_rc(im) \
    for (int r = 0; r < im.rows; ++r) \
    for (int c = 0; c < im.cols; ++c)

struct Neighbor
{
    int dr, dc;
    uint8_t bitmask;
    
    // Corresponding bitmask in the neighbor. For example if my neighbor is (-1, 0) with bitmask 0b10, 
    // then I am neighbor (1,0) for that pixel, and the bitmask for me in that pixel is 0b01000000.
    uint8_t reprocBitmask;
};

constexpr std::array<Neighbor, 8> neighbors3x3 = {
    Neighbor{-1, -1, 0b00000001, 0b10000000},
    Neighbor{-1,  0, 0b00000010, 0b01000000},
    Neighbor{-1,  1, 0b00000100, 0b00100000},
    Neighbor{ 0, -1, 0b00001000, 0b00010000},
    Neighbor{ 0,  1, 0b00010000, 0b00001000},
    Neighbor{ 1, -1, 0b00100000, 0b00000100},
    Neighbor{ 1,  0, 0b01000000, 0b00000010},
    Neighbor{ 1,  1, 0b10000000, 0b00000001},
};

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
        std::cerr << "Usage: parse_chart_v1 input_image" << std::endl;
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

    cv::Mat1i labels (im.cols, im.rows);
    labels = -1; // no labels by default.

    // Assign the 0 label to background pixels.
    for_all_rc (im)
    {
        if (bgMask(r,c))
            labels(r,c) = 0;
    }

    cv::Mat1b connectivityBitmask (im.cols, im.rows);

    // FIXME: here we meant to solve for alpha and the color in the
    // entire neighorhood to detect inconsistent values of alpha.
    // But keeping it this way for now for simplicity.
    for_all_rc (im)
    {
        const cv::Vec3b currRgb = im(r,c);
        const cv::Vec3b currBgColor = bgColorIm(r,c);
        uint8_t bitmask = 0;
        for (const auto& neighb : neighbors3x3)
        {
            const int otherR = r+neighb.dr;
            const int otherC = c+neighb.dc;
            const cv::Vec3b otherRgb = im(otherR, otherC);
            int d = std::get<0>(alphaAwareDist(currRgb, otherRgb, currBgColor));
            if (d < 10)
            {
                bitmask |= neighb.bitmask;
            }
        }
        connectivityBitmask(r,c) = bitmask;
    }

    imshow("connectivityBitmaskStep1", connectivityBitmask);
    
    // Finalize the connectivity by only connecting pixels that manually agree
    // that they are connected. Actually useless right now because the
    // neighborhood check is symmetric, but this is going to change later
    // since the entire 3x3 neighboorhood will be used.
    for_all_rc (im)
    {
        uint8_t bitmask = connectivityBitmask(r,c);
        for (const auto& neighb : neighbors3x3)
        {
            if (bitmask & neighb.bitmask)
            {
                uint8_t neighbBitmask = connectivityBitmask(r+neighb.dr,c+neighb.dc);
                // If the neighbor is not connected to me, then disconnect myself from it,
                if (!(neighbBitmask & neighb.reprocBitmask))
                {
                    bitmask ^= neighb.bitmask;
                    neighbBitmask ^= connectivityBitmask(r+neighb.dr,c+neighb.dc);
                }
            }
        }
        connectivityBitmask(r,c) = bitmask;
    }
    
    imshow("connectivityBitmaskSymmetric", connectivityBitmask);
    
    cv::waitKey(1);

    // Propagation of the connected pixels.

    cv::Point lastSeedPoint (0,0);
    
    // 0 is taken for the background.
    int nextLabel = 1;
    
    std::unordered_map<int,cv::Vec3b> labelColorMap;
    labelColorMap[-1] = cv::Vec3b(0,0,0);
    labelColorMap[0] = bgColor;

    while (true)
    {
        cv::Point nextSeedPoint (-1, -1);
        {
            bool found = false;
            // Find the next seed point
            for (int r = lastSeedPoint.y; !found && r < im.rows; ++r)
            for (int c = lastSeedPoint.x; !found && c < im.cols; ++c)
            {
                if (labels(r,c) == -1)
                {
                    nextSeedPoint = cv::Point(c,r);
                    found = true;
                }
            }
        }

        fprintf(stderr, "nextSeedPoint = %d %d\n", nextSeedPoint.x, nextSeedPoint.y);
        
        // No more seedpoints
        if (nextSeedPoint.x < 0)
            break;

        // Bookkeeping before we forget.
        lastSeedPoint = nextSeedPoint;

        const int currLabel = nextLabel;
        ++nextLabel;
        labelColorMap[currLabel] = cv::Vec3b(rand()%255, rand()%255, rand()%255);

        std::deque<cv::Point> pointsToProcess;
        pointsToProcess.push_back(nextSeedPoint);        

        while (!pointsToProcess.empty())
        {
            const cv::Point currP = pointsToProcess.front();
            pointsToProcess.pop_front();

            const int r = currP.y;
            const int c = currP.x;
            const cv::Vec3b currRgb = im(r,c);
            
            labels(r, c) = currLabel;

            uint8_t connectedBits = connectivityBitmask(r,c);

            for (const auto& neighb : neighbors3x3)
            {
                const int otherR = r + neighb.dr;
                const int otherC = c + neighb.dc;

                // Already labelled?
                if (labels(otherR, otherC) != -1)
                    continue;
                
                if (connectedBits & neighb.bitmask)
                {
                    labels(otherR, otherC) = currLabel;
                    pointsToProcess.push_back(cv::Point(otherC, otherR));
                }
            }
        }
    }

    cv::Mat3b labelsAsColors (labels.rows, labels.cols);
    for_all_rc(labelsAsColors)
    {
        labelsAsColors(r,c) = labelColorMap[labels(r,c)];
    }
    
    imshow("labels", labelsAsColors);
    cv::waitKey();
    
    return 0;
}
