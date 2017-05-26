#include "feature_patches.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace planar_tracking {

FeaturePatches::FeaturePatches() {}

FeaturePatches::~FeaturePatches() {}

bool FeaturePatches::Setup(const cv::Mat& image,
                           int numIdealPoints,
                           const cv::Mat& mask,
                           const int binSize,
                           cv::Size subPixWinSize,
                           double qualityLevel,
                           double minDistance,
                           int blockSize,
                           bool useHarrisDetector,
                           double k) {
  bool status = true;
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(image, points, MAX_COUNT, qualityLevel, minDistance,
                          mask, blockSize, useHarrisDetector, k);
  const int numFoundPoints = static_cast<int>(points.size());
  if (numFoundPoints < MIN_COUNT) {
    return false;
  }
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                            20, 0.03);
  cv::cornerSubPix(image, points, subPixWinSize, cv::Size(-1, -1), termcrit);
  char *pointStatus = new char[numFoundPoints];
  memset(pointStatus, 1, numFoundPoints * sizeof(char));
  int numGoodPoints = FindGoodPoints(image, points, pointStatus, binSize);
  int numPoints = std::min(numGoodPoints, numIdealPoints);
  if (numPoints < MIN_COUNT) {
    status = false;
    return status;
  }

  int m = 0;
  points_2d_.resize(numPoints);
  for (int i = 0; i < numFoundPoints; ++i) {
    if (pointStatus[i] > 0 && m < numPoints) {
      points_2d_[m].x = points[i].x;
      points_2d_[m].y = points[i].y;
      ++m;
    }
  }
  if (m < numPoints) {
    points_2d_.erase(points_2d_.begin() + m, points_2d_.end());
  }
  delete [] pointStatus;
  return status;
}

bool FeaturePatches::Setup(const std::vector<cv::Point2f>& points_2d) {
  bool status = true;
  if (points_2d.size() == 0) {
    status = false;
    return status;
  }
  points_2d_ = points_2d;
  return status;
}

int FeaturePatches::FindGoodPoints(const cv::Mat& image,
                                   const std::vector<cv::Point2f>& points,
                                   char *pointStatus,
                                   const int binSize,
                                   const int numBinPoints) {
  int numGoodPoints = 0;
  int width = image.cols;
  int height = image.rows;
  int xBins = static_cast<int>(static_cast<float>(width + binSize - 1) /
                               static_cast<float>(binSize));
  int yBins = static_cast<int>(static_cast<float>(height + binSize - 1) /
                               static_cast<float>(binSize));
  int numBins = xBins * yBins;
  std::vector<int> hist2d_host(numBins, 0);
  int numPoints = points.size();
  for (int i = 0; i < numPoints; ++i) {
    if (pointStatus[i] > 0) {
      int pos = GetHist2dPos(points[i].x, points[i].y, xBins, binSize);
      if (hist2d_host[pos] <= numBinPoints) {
        hist2d_host[pos]++;
        numGoodPoints++;
      } else {
        pointStatus[i] = -1;
      }
    }
  }
  return numGoodPoints;
}

}  // namespace planar_tracking
