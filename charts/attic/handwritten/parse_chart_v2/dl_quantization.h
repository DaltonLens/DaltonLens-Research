#pragma once

#include <deque>

#include <glfw_opencv/imgui_cvlog_gl_opencv.h>

// 256 colors max.

inline int maxIndex(cv::Vec3i v) 
{
    if (v[1] >= v[2])
        return (v[1] > v[0]) ? 1 : 0;   
    return v[0] >= v[2] ? 0 : 2;
}

class MedianCut
{
public:
    void apply(const cv::Mat3b &im_rgb,
               const cv::Mat1b& mask,
               int maxColors,
               cv::Mat1b &indexed,
               std::vector<cv::Vec3b> &palette_rgb)
    {
        assert (maxColors <= 256);

        cv::Mat3b im;
        cv::cvtColor(im_rgb, im, cv::COLOR_BGR2Lab);
        
        Bucket firstBucket;
        firstBucket.pixels.resize (im.rows*im.cols);
        for_all_rc (im)
        {
            // Was playing with different weights, ended up with just 1.0.
            if (mask(r,c))
                firstBucket.pixels[r*im.cols+c] = RgbAndIndexAndWeight { im(r,c), r*im.cols+c, 1.0f };
        }

        computeRgbRangesAndWeights (firstBucket);
        
        std::priority_queue<Bucket,
                            std::vector<Bucket>,
                            CompareBucketByWeightedRange> bucketQueue;
        bucketQueue.push (firstBucket);

        // We'll keep the frozen buckets separately to avoid splitting them again and again
        // and leave room for more splits.
        std::set<Bucket> finalizedBuckets;
        while (!bucketQueue.empty() && (finalizedBuckets.size() + bucketQueue.size()) < maxColors-1)
        {
            Bucket worstBucket = bucketQueue.top();
            bucketQueue.pop();
            
            fprintf (stderr, "Queue: popping value with range = %d\n", worstBucket.maxRange());
            
            Bucket leftBucket, rightBucket;
            splitBucket (worstBucket, leftBucket, rightBucket);
            
            auto enqueueBucket = [&](const Bucket& bucket)
            {
                bool isFinal = bucket.isUniform() || bucket.maxRange() < 5;
                if (isFinal)
                    finalizedBuckets.insert(bucket);
                else
                    bucketQueue.push(bucket);
            };
            
            if (rightBucket.pixels.empty())
            {
                finalizedBuckets.insert(leftBucket);
            }
            else
            {
                enqueueBucket (leftBucket);
                enqueueBucket (rightBucket);
            }
        }
              
        while (!bucketQueue.empty())
        {
            finalizedBuckets.insert(bucketQueue.top());
            bucketQueue.pop();
        }
                
        std::set<cv::Vec3b> uniqueColors;

        for (const auto& bucket : finalizedBuckets)
        {
            fprintf (stderr, "[Bucket] range = %d %d %d (%d pixels)\n",
                     bucket.rgbRanges[0],
                     bucket.rgbRanges[1],
                     bucket.rgbRanges[2],
                     (int)bucket.pixels.size());
            
            cv::Vec3f cumulatedRgb (0,0,0);
            for (const auto& rgbAndIndex : bucket.pixels)
            {
                cumulatedRgb += cv::Vec3f(rgbAndIndex.rgb[0], rgbAndIndex.rgb[1], rgbAndIndex.rgb[2]);
            }
            cumulatedRgb *= 1.0f/bucket.pixels.size();
            
            cv::Vec3b newColor;
            for (int k = 0; k < 3; ++k)
            {
                newColor[k] = uint8_t(int(cumulatedRgb[k] + 0.5f));
            }

            uniqueColors.insert(newColor);
        }
        
        std::vector<cv::Vec3b> palette;
        palette.clear();
        palette.insert(palette.end(), uniqueColors.begin(), uniqueColors.end());
        indexed.create(im.rows, im.cols);
        
        std::map<int, int> errorHistogram;
        
        for_all_rc (im)
        {
            cv::Vec3i rgb = im(r,c);
            float min_d = std::numeric_limits<float>::max();
            int best_i = -1;
            for (int i = 0; i < palette.size(); ++i)
            {
                float d = cv::norm((cv::Vec3i)palette[i] - rgb, cv::NORM_L1);
                if (d < min_d)
                {
                    min_d = d;
                    best_i = i;
                }
            }
            indexed(r,c) = best_i;
            errorHistogram[min_d] += 1;
            
            //            if (min_d > 5)
            //            {
            //                fprintf (stderr, "high dist [%d %d %d] <-> [%d %d %d] (d=%f)\n",
            //                         rgb[0],
            //                         rgb[1],
            //                         rgb[2],
            //                         palette[best_i][0],
            //                         palette[best_i][1],
            //                         palette[best_i][2],
            //                         min_d);
            //            }
        }
        
        palette_rgb.clear();
        cv::cvtColor (palette, palette_rgb, cv::COLOR_Lab2BGR);
        
        for (const auto& it : errorHistogram)
        {
            fprintf (stderr, "%d -> %d\n", it.first, it.second);
        }
    }

private:
    struct RgbAndIndexAndWeight
    {
        cv::Vec3b rgb;
        int index;
        float weight;
    };
    struct Bucket
    {
        std::vector<RgbAndIndexAndWeight> pixels;
        double cumulatedGradientWeight = NAN;
        
        cv::Vec3i rgbRanges = cv::Vec3i(-1,-1,-1);
        cv::Vec3i rgbMin = cv::Vec3i(-1,-1,-1);
        cv::Vec3i rgbMax = cv::Vec3i(-1,-1,-1);
        
        int maxRange() const
        {
            return std::max(std::max(rgbRanges[0], rgbRanges[1]), rgbRanges[2]);
        }
        
        bool isUniform() const
        {
            return rgbMin == rgbMax;
        }
        
        // Same if rgbMin and rgbMax are the same.
        bool operator< (const Bucket& rhs) const
        {
            if (rgbMin < rhs.rgbMin)
                return true;
            
            if (rhs.rgbMin < rgbMin)
                return false;
            
            return rgbMax < rhs.rgbMax;
        }
    };
    
    struct CompareBucketByWeightedRange
    {
        bool operator()(const Bucket& lhs, const Bucket& rhs) const
        {
            // return lhs.maxRange()*sqrt(lhs.pixels.size()) < rhs.maxRange()*sqrt(rhs.pixels.size());
            return lhs.maxRange()*sqrt(lhs.cumulatedGradientWeight) < rhs.maxRange()*sqrt(rhs.cumulatedGradientWeight);
        }
    };

private:
    void splitBucket (Bucket& bucket,
                      Bucket& lhsBucket,
                      Bucket& rhsBucket) const
    {
        fprintf (stderr, "rgb ranges %d %d %d\n",
                 bucket.rgbRanges[0],
                 bucket.rgbRanges[1],
                 bucket.rgbRanges[2]);
        
        int maxRangeChannel = maxIndex(bucket.rgbRanges);
        
        sortIndicesByChannel(bucket.pixels, maxRangeChannel);

        fprintf (stderr, "first pixel = %d %d %d\n",
                 bucket.pixels.front().rgb[0],
                 bucket.pixels.front().rgb[1],
                 bucket.pixels.front().rgb[2]);
        
        fprintf (stderr, "last pixel = %d %d %d\n",
                 bucket.pixels.back().rgb[0],
                 bucket.pixels.back().rgb[1],
                 bucket.pixels.back().rgb[2]);
        
        int medianIndex = bucket.pixels.size() / 2;
        cv::Vec3b lastValueOfFirstSet = bucket.pixels[medianIndex-1].rgb;
        while (medianIndex < bucket.pixels.size()
               && bucket.pixels[medianIndex].rgb == lastValueOfFirstSet)
        {
            ++medianIndex;
        }
                
        lhsBucket.pixels.resize(medianIndex);
        std::copy(bucket.pixels.begin(), bucket.pixels.begin() + medianIndex, lhsBucket.pixels.begin());
        computeRgbRangesAndWeights (lhsBucket);
        
        if (medianIndex < bucket.pixels.size())
        {
            fprintf (stderr, "median pixel = %d %d %d\n",
                     bucket.pixels[medianIndex].rgb[0],
                     bucket.pixels[medianIndex].rgb[1],
                     bucket.pixels[medianIndex].rgb[2]);
            
            rhsBucket.pixels.resize(bucket.pixels.size() - medianIndex);
            std::copy(bucket.pixels.begin() + medianIndex, bucket.pixels.end(), rhsBucket.pixels.begin());
            computeRgbRangesAndWeights (rhsBucket);
        }
    }
    
    template <unsigned k1, unsigned k2, unsigned k3>
    void sortIndicesByChannel (std::vector<RgbAndIndexAndWeight>& pixels) const
    {
        std::sort (pixels.begin(), pixels.end(), [](const RgbAndIndexAndWeight& lhs, const RgbAndIndexAndWeight& rhs) {
            if (lhs.rgb[k1] < rhs.rgb[k1])
                return true;
            
            if (lhs.rgb[k1] > rhs.rgb[k1])
                return false;
            
            if (lhs.rgb[k2] < rhs.rgb[k2])
                return true;
            
            if (lhs.rgb[k2] > rhs.rgb[k2])
                return false;
            
            return lhs.rgb[k3] < rhs.rgb[k3];
        });
    }

    void sortIndicesByChannel (std::vector<RgbAndIndexAndWeight>& pixels, int channel) const
    {
        switch (channel)
        {
            case 0: sortIndicesByChannel<0,1,2>(pixels); break;
            case 1: sortIndicesByChannel<1,0,2>(pixels); break;
            case 2: sortIndicesByChannel<2,0,1>(pixels); break;
            default: assert(false); break;
        }
    }

    void computeRgbRangesAndWeights(Bucket& bucket) const
    {
        cv::Vec3i rgbMin (255,255,255);
        cv::Vec3i rgbMax (0,0,0);
        bucket.cumulatedGradientWeight = 0.;
        for (const auto& rgbAndIndex : bucket.pixels)
        {
            bucket.cumulatedGradientWeight += rgbAndIndex.weight;
            
            for (int k = 0; k < 3; ++k)
            {
                rgbMin[k] = std::min(rgbMin[k], (int)rgbAndIndex.rgb[k]);
                rgbMax[k] = std::max(rgbMax[k], (int)rgbAndIndex.rgb[k]);
            }
        }

        bucket.rgbMin = rgbMin;
        bucket.rgbMax = rgbMax;
        bucket.rgbRanges = rgbMax - rgbMin;
    }
};

cv::Mat3b indexedToRgb(const cv::Mat1b &indexed, const std::vector<cv::Vec3b> &palette)
{
    cv::Mat3b out (indexed.rows, indexed.cols);
    for_all_rc (indexed)
    {
        out(r,c) = palette[indexed(r,c)];
    }
    return out;
}
