#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include "feature_patches.h"

#include <opencv2/core/core.hpp>
#include <vector>

namespace planar_tracking {

class KeyFrame {
 public:
  KeyFrame();
  ~KeyFrame();

  bool Setup(const cv::Mat& keyframe_image,
             const cv::Size2d& template_size,
             const std::vector<cv::Point2f>& template_contours,
             const cv::Size& win_size = cv::Size(21, 21));

  void InitTwoChannelsComplex(cv::Mat* I,
                              const cv::Mat& table);
  void ExtractMask(const cv::Mat& I, cv::Mat* mask);
  cv::Mat GetTemplateMask(const cv::Mat& srcImg,
                          cv::Rect template_img_rect,
                          const std::vector<cv::Point2f>& contour_pts,
                          uchar frgValue = 255,
                          uchar bkgValue = 0);
  cv::Rect GetTemplateBoundingRect(
    cv::Size img_size,
    const std::vector<cv::Point2f>& contour_pts,
    cv::Size win_size);

  FeaturePatches feature_patches_;
  cv::Mat template_mask_;
  cv::Mat template_roi_;
  cv::Rect template_rect_;
  std::vector<cv::Point2f> template_contours_;
  cv::Mat template_homography_;
  cv::Mat template_warp_;
  cv::Size win_size_;
  cv::Size2d template_size_;
  std::vector<cv::Point2f> template_corners_;
  std::vector<cv::Point3f> template_points_3d_;
};

}  // namespace planar_tracking

#endif  // KEYFRAME_H_
