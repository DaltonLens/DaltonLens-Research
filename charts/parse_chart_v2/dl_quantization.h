#pragma once

#include "dl_opencv.h"

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
    void apply(const cv::Mat3b &im,
               int maxColors,
               cv::Mat1b &indexed,
               std::vector<cv::Vec3b> &palette)
    {
        assert (maxColors <= 256);

        palette.clear();

        Bucket firstBucket;
        firstBucket.pixels.resize (im.rows*im.cols);
        for_all_rc (im)
        {
            firstBucket.pixels[r*im.cols+c] = std::make_pair(im(r,c), r*im.cols+c);
        }

        computeRgbRanges (firstBucket);
        
        std::vector<Bucket> buckets;
        buckets.push_back (firstBucket);

        // We'll keep the frozen buckets separately to avoid splitting them again and again
        // and leave room for more splits. We don't know in advance whether all the buckets
        // will get split since some might have a zero-range.
        std::set<Bucket> uniformBuckets;
        while ((uniformBuckets.size() + buckets.size()) <= maxColors)
        {
            buckets = splitBuckets(buckets, uniformBuckets);
        }
        
        // Merge back the uniform guys. Put them first to make sure we don't kill them after
        // if we added too many colors.
        buckets.insert(buckets.begin(), uniformBuckets.begin(), uniformBuckets.end());
        
        // Remove the most recent splits if we went too far.
//        while (buckets.size() > maxColors)
//            buckets.pop_back();
        
        std::set<cv::Vec3b> uniqueColors;

        for (int i = 0; i < buckets.size(); ++i)
        {
            fprintf (stderr, "[%d] range = %d %d %d (%d pixels)\n",
                     i,
                     buckets[i].rgbRanges[0],
                     buckets[i].rgbRanges[1],
                     buckets[i].rgbRanges[2],
                     (int)buckets[i].pixels.size());
            
            cv::Vec3f cumulatedRgb (0,0,0);
            for (const auto& rgbAndIndex : buckets[i].pixels)
            {
                cumulatedRgb += cv::Vec3f(rgbAndIndex.first[0], rgbAndIndex.first[1], rgbAndIndex.first[2]);
            }
            cumulatedRgb *= 1.0f/buckets[i].pixels.size();
            
            cv::Vec3b newColor;
            for (int k = 0; k < 3; ++k)
            {
                newColor[k] = uint8_t(int(cumulatedRgb[k] + 0.5f));
            }

            uniqueColors.insert(newColor);
        }
        
        palette.clear();
        palette.insert(palette.end(), uniqueColors.begin(), uniqueColors.end());
        indexed.create(im.rows, im.cols);
        
        for_all_rc (im)
        {
            cv::Vec3i rgb = im(r,c);
            float min_d = FLT_MAX;
            int best_i = -1;
            for (int i = 0; i < palette.size(); ++i)
            {
                float d = cv::norm((cv::Vec3i)palette[i] - rgb);
                if (d < min_d)
                {
                    min_d = d;
                    best_i = i;
                }
            }
            indexed(r,c) = best_i;
        }
    }

private:
    using RgbAndIndex = std::pair<cv::Vec3b, int>;
    struct Bucket
    {
        std::vector<RgbAndIndex> pixels;
        cv::Vec3i rgbRanges = cv::Vec3i(-1,-1,-1);
        cv::Vec3i rgbMin = cv::Vec3i(-1,-1,-1);
        cv::Vec3i rgbMax = cv::Vec3i(-1,-1,-1);
        
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

private:
    std::vector<Bucket> splitBuckets (std::vector<Bucket>& inputBuckets,
                                      std::set<Bucket>& uniformBuckets) const
    {
        std::vector<Bucket> outputBuckets;

        for (auto& bucket : inputBuckets)
        {
            fprintf (stderr, "rgb ranges %d %d %d\n",
                     bucket.rgbRanges[0],
                     bucket.rgbRanges[1],
                     bucket.rgbRanges[2]);
            
            int maxRangeChannel = maxIndex(bucket.rgbRanges);
                        
            sortIndicesByChannel(bucket.pixels, maxRangeChannel);

            fprintf (stderr, "first pixel = %d %d %d\n",
                     bucket.pixels.front().first[0],
                     bucket.pixels.front().first[1],
                     bucket.pixels.front().first[2]);
            
            fprintf (stderr, "last pixel = %d %d %d\n",
                     bucket.pixels.back().first[0],
                     bucket.pixels.back().first[1],
                     bucket.pixels.back().first[2]);
            
            Bucket lhsBucket, rhsBucket;
            int medianIndex = bucket.pixels.size() / 2;
            
            fprintf (stderr, "median pixel = %d %d %d\n",
                     bucket.pixels[medianIndex].first[0],
                     bucket.pixels[medianIndex].first[1],
                     bucket.pixels[medianIndex].first[2]);
            
            lhsBucket.pixels.resize(medianIndex);
            rhsBucket.pixels.resize(bucket.pixels.size() - medianIndex);
            std::copy(bucket.pixels.begin(), bucket.pixels.begin() + medianIndex, lhsBucket.pixels.begin());
            std::copy(bucket.pixels.begin() + medianIndex, bucket.pixels.end(), rhsBucket.pixels.begin());

            computeRgbRanges (lhsBucket);
            computeRgbRanges (rhsBucket);
            
            if (lhsBucket.isUniform())
                uniformBuckets.insert (lhsBucket);
            else
                outputBuckets.push_back (lhsBucket);
            
            if (rhsBucket.isUniform())
                uniformBuckets.insert (rhsBucket);
            else
                outputBuckets.push_back (rhsBucket);
        }

        return outputBuckets;
    }
    
    template <unsigned k>
    void sortIndicesByChannel (std::vector<RgbAndIndex>& pixels) const
    {
        std::sort (pixels.begin(), pixels.end(), [](const RgbAndIndex& lhs, const RgbAndIndex& rhs) {
                    return lhs.first[k] < rhs.first[k];
        });
    }

    void sortIndicesByChannel (std::vector<RgbAndIndex>& pixels, int channel) const
    {
        switch (channel)
        {
            case 0: sortIndicesByChannel<0>(pixels); break;
            case 1: sortIndicesByChannel<1>(pixels); break;
            case 2: sortIndicesByChannel<2>(pixels); break;
            default: assert(false); break;
        }
    }

    void computeRgbRanges(Bucket& bucket) const
    {
        cv::Vec3i rgbMin (255,255,255);
        cv::Vec3i rgbMax (0,0,0);
        for (const auto& rgbAndIndex : bucket.pixels)
        {
            for (int k = 0; k < 3; ++k)
            {
                rgbMin[k] = std::min(rgbMin[k], (int)rgbAndIndex.first[k]);
                rgbMax[k] = std::max(rgbMax[k], (int)rgbAndIndex.first[k]);
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
