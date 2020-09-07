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
#include "Utils.h"

#include <imgui_cvlog.h>
#include <glfw_opencv/imgui_cvlog_gl_opencv.h>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/icm.hxx>

#include <unordered_map>
#include <utility>
#include <cstdio>
#include <thread>
#include <fstream>

using namespace cv;
using namespace opengm;

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

int runProcessing (int argc, char** argv)
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
    
    assert (std::ifstream (argv[1]).good());
    
    cv::Mat3b im = imread(argv[1]);
    assert (im.data);

    ImGui::CVLog::UpdateImage("im", im);
    
//    {
//        static MouseAction im_onMouse = [im]( int event, int x, int y, int ) {
//            if (event != EVENT_LBUTTONDOWN)
//                return;
//
//            auto bgr = im(y,x);
//            fprintf (stderr, "[%d %d] -> [%d %d %d]", x, y, bgr[2], bgr[1], bgr[0]);
//            
//            cv::Vec3b bgColor (135, 27, 107);
//
//            for (int dr = -1; dr <= 1; ++dr)
//            for (int dc = -1; dc <= 1; ++dc)
//            {
//                auto neighb = im(y+dr, x+dc);
//
//                int dist; float alpha1, alpha2;
//                std::tie(dist, alpha1, alpha2) = alphaAwareDist(bgr,
//                                                                neighb,
//                                                                bgColor);
//
//                fprintf (stderr, "\t[%d %d] -> %d (%.2f %.2f)\n",
//                         dr, dc,
//                         dist, alpha1, alpha2);
//            }
//        };
//        setMouseAction( "im", im_onMouse );
//        cv::waitKey(0);
//    }

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

    ImGui::CVLog::UpdateImage("lowGradPixels", lowGradPixels);
    ImGui::CVLog::UpdateImage("erodedLowGradPixels", erodedLowGradPixels);

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
    cv::Mat3b bgColorIm (im.rows, im.cols);
    bgColorIm = bgColor;
        
    ImGui::CVLog::UpdateImage("bgColorIm", bgColorIm);

    cv::Mat1b bgMask (im.rows, im.cols);
    bgMask = 0;
    for_all_rc (im)
    {
        if (rgbDist(bgColorIm(r,c), im(r,c)) < 10)
          bgMask(r,c) = 255;  
    }
    
    // Ignore gray pixels as color-blind people see them properly.
    cv::Mat1b fgMask (im.rows, im.cols);
    for_all_rc (im)
    {
        const int drg = std::abs(im(r,c)[0] - im(r,c)[1]);
        const int drb = std::abs(im(r,c)[0] - im(r,c)[2]);
        const int dgb = std::abs(im(r,c)[1] - im(r,c)[2]);
        bool isColor = std::max(drg, std::max(drb, dgb)) > 10;
        fgMask(r,c) = isColor ? 255 : 0;
    }

    ImGui::CVLog::UpdateImage("bgMask", bgMask);
    ImGui::CVLog::UpdateImage("fgMask", fgMask);

    cv::Mat1f alphaImage (im.rows, im.cols);
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
    const int numColors = 16;
    quantizer.apply (im, fgMask, numColors, indexed, palette);
    
    cv::Mat3b indexedAsRgb = indexedToRgb(indexed, palette);
    for_all_rc(indexedAsRgb)
    {
        if (!fgMask(r,c))
            indexedAsRgb(r,c) = im(r,c);
    }

    fprintf (stderr, "Palette: \n");
    for (const auto& rgb : palette)
        fprintf (stderr, "\t[%d %d %d]\n", rgb[0], rgb[1], rgb[2]);
    
    ImGui::CVLog::UpdateImage("indexed", indexed);
    ImGui::CVLog::UpdateImage("quantized", indexedAsRgb);

    // model parameters (global variables are used only in example code)
    const size_t nx = im.cols; // width of the grid
    const size_t ny = im.rows; // height of the grid
    const size_t numberOfLabels = numColors + 1; // 0 is a special label for pixels to skip touch.
    double lambda = 0.1; // coupling strength of the Potts model

    // this function maps a node (x, y) in the grid to a unique variable index
    auto variableIndex = [nx](const size_t x, const size_t y) {
        return x + nx * y; 
    };

    // construct a label space with
    // - nx * ny variables
    // - each having numberOfLabels many labels
    typedef SimpleDiscreteSpace<size_t, size_t> Space;
    Space space(nx * ny, numberOfLabels);

    // construct a graphical model with
    // - addition as the operation (template parameter Adder)
    // - support for Potts functions (template parameter PottsFunction<double>)
    typedef GraphicalModel<double, Adder, OPENGM_TYPELIST_2(ExplicitFunction<double>, PottsFunction<double>), Space> Model;
    Model gm(space);

    // for each node (x, y) in the grid, i.e. for each variable
    // variableIndex(x, y) of the model, add one 1st order functions
    // and one 1st order factor
    for (size_t y = 0; y < ny; ++y)
    for (size_t x = 0; x < nx; ++x)
    {
        // function
        const size_t shape[] = {numberOfLabels};
        ExplicitFunction<double> f(shape, shape + 1);
        for (size_t s = 0; s < numberOfLabels; ++s)
        {
            f(s) = (1.0 - lambda) * rand() / RAND_MAX;
        }
        Model::FunctionIdentifier fid = gm.addFunction(f);

        // factor
        size_t variableIndices[] = {variableIndex(x, y)};
        gm.addFactor(fid, variableIndices, variableIndices + 1);
    }
    
    // add one (!) 2nd order Potts function
    PottsFunction<double> f(numberOfLabels, numberOfLabels, 0.0, lambda);
    Model::FunctionIdentifier fid = gm.addFunction(f);
    
    // for each pair of nodes (x1, y1), (x2, y2) which are adjacent on the grid,
    // add one factor that connects the corresponding variable indices and
    // refers to the Potts function
    for(size_t y = 0; y < ny; ++y)
    for(size_t x = 0; x < nx; ++x)
    {
        if (x + 1 < nx)
        {
            // (x, y) -- (x + 1, y)
            size_t variableIndices[] = {variableIndex(x, y), variableIndex(x + 1, y)};
            std::sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
        if (y + 1 < ny)
        {
            // (x, y) -- (x, y + 1)
            size_t variableIndices[] = {variableIndex(x, y), variableIndex(x, y + 1)};
            std::sort(variableIndices, variableIndices + 2);
            gm.addFactor(fid, variableIndices, variableIndices + 2);
        }
    }
    
    // set up the optimizer (loopy belief propagation)
    typedef BeliefPropagationUpdateRules<Model, opengm::Minimizer> UpdateRules;
    typedef MessagePassing<Model, opengm::Minimizer, UpdateRules, opengm::MaxDistance> BeliefPropagation;
    const size_t maxNumberOfIterations = 1;
    const double convergenceBound = 1e-7;
    const double damping = 0.5;
    BeliefPropagation::Parameter parameter(maxNumberOfIterations, convergenceBound, damping);
    BeliefPropagation bp(gm, parameter);
    
    // optimize (approximately)
    {
        dl::ScopeTimer _ ("Inference");
        BeliefPropagation::VerboseVisitorType visitor;
        bp.infer(visitor);
    }

    // obtain the (approximate) argmin
    std::vector<size_t> labeling(nx * ny);
    bp.arg(labeling);
    
    // output the (approximate) argmin
    //    {
    //        size_t varIdx = 0;
    //        for(size_t y = 0; y < ny; ++y)
    //        {
    //            for(size_t x = 0; x < nx; ++x)
    //            {
    //                std::cerr << labeling[varIdx] << ' ';
    //                ++varIdx;
    //            }
    //            std::cerr << std::endl;
    //        }
    //    }

    return 0;
}

int main (int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: parse_chart_v2 input_image" << std::endl;
        return 1;
    }

    ImGui::CVLog::OpenCVGLWindow window;
    window.initializeContexts("Parse chart v2", 1980, 1080);

    std::thread processingThread ([&]() { runProcessing(argc, argv); });

    window.run();
    processingThread.join();
    window.shutDown();    

    return 0;
}
