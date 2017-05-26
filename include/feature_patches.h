#ifndef FEATURE_PATCHES_H_
#define FEATURE_PATCHES_H_

#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

namespace planar_tracking {

class FeaturePatches {
 public:
  FeaturePatches();
  ~FeaturePatches();

  bool Setup(const cv::Mat& image, int numIdealPoints,
             const cv::Mat& mask = cv::Mat(), const int binSize = 20,
             cv::Size subPixWinSize = cv::Size(5, 5),
             double qualityLevel = 0.005f, double minDistance = 10.0f,
             int blockSize = 3, bool useHarrisDetector = false,
             double k = 0.04);
  bool Setup(const std::vector<cv::Point2f>& points_2d);
  inline std::vector<cv::Point2f>* FeaturesPoints() {
    return &points_2d_;
  }
  inline const std::vector<cv::Point2f>* FeaturesPoints() const {
    return &points_2d_;
  }

  inline unsigned int FeaturesPointsCount() const {
    return points_2d_.size();
  }

  inline cv::Point2f* FeaturesPoints(unsigned int index) {
    if (index >= points_2d_.size()) {
      return nullptr;
    }
    return &(points_2d_[index]);
  }

  inline int GetHist2dPos(float x, float y, int binWidth, int binSize) {
    int c = static_cast<int>(floor(x / static_cast<float>(binSize)));
    int r = static_cast<int>(floor(y / static_cast<float>(binSize)));
    return (r * binWidth + c);
  }

  static const int MAX_COUNT = 50;
  static const int MIN_COUNT = 15;

 protected:
  int FindGoodPoints(const cv::Mat &image, const std::vector<cv::Point2f>& points,
                     char *pointStatus, const int binSize, const int numBinPoints = 1);
  void Cleanup();
  std::vector<cv::Point2f> points_2d_;
};

}  // namespace planar_tracking

#endif  // FEATURE_PATCHES_H_
