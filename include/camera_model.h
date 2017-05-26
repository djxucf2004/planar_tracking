#ifndef CAMERA_MODEL_H_
#define CAMERA_MODEL_H_

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace planar_tracking {

class CameraModel {
 public:
  CameraModel(const std::string& calib_filename);
  ~CameraModel();

  // Undistort the input image
  bool Undistort(const cv::Mat& image_in, cv::Mat* image_out,
                 const int pyr_level = 0) const;
  cv::Point2f Project(const cv::Vec3f &point) const;
  cv::Point2d Project(const cv::Vec3d &point) const;

  // maximum pyramid levels
  int pyr_levels_max_;
  // Camera intrinsics
  std::vector<cv::Mat> camera_matrix_pyr_;
  // Lens distortion coefficients
  cv::Mat camera_distortions_;
  // Input image size
  std::vector<cv::Size> image_size_pyr_;
};

}  // namespace planar_tracking

#endif  // CAMERA_MODEL_H_
