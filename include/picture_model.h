#ifndef PICTURE_MODEL_H_
#define PICTURE_MODEL_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace planar_tracking {

// Planar picture model for tracking
class PictureModel {
 public:
  explicit PictureModel(const std::string& filename);
  ~PictureModel() {}

  cv::Mat image_;
  std::string name_;
  // Picture origin position
  cv::Point2d picture_origin_;
  // Meters per pixel
  double meters_per_pixel_;
  // Picture size (mm)
  cv::Size2d picture_size_;
  // Picture image size (pixels)
  cv::Size image_size_;
  // Picture contour corner points in 2d (pixels)
  std::vector<cv::Point2f> corner_pts_2d_;
  // Picture contour corner points in 3d (mm)
  std::vector<cv::Point3f> corner_pts_3d_;
};

}  // namespace planar_tracking

#endif  // PICTURE_MODEL_H_
