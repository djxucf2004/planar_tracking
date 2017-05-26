#include "camera_model.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

namespace planar_tracking {

// Constructor
CameraModel::CameraModel(const std::string& calib_filename)
  : pyr_levels_max_(1) {
  // Load the camera intrinsics
  cv::FileStorage fs;
  fs.open(calib_filename.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Unable to open the camera calibration file!" << std::endl;
    return;
  }

  // pyramid cameras
  camera_matrix_pyr_.resize(pyr_levels_max_ + 1);
  image_size_pyr_.resize(pyr_levels_max_ + 1);

  // read the original camera intrinsics for pyramid level = 0
  cv::FileNode fn = fs["calibration"];
  fn["camera_matrix"] >> camera_matrix_pyr_[0];
  std::cout << "camera matrix = " << camera_matrix_pyr_[0];
  fn["distortion"] >> camera_distortions_;
  std::cout << "camera_distortions = " << camera_distortions_;
  image_size_pyr_[0].width = static_cast<int>(fn["width"]);
  std::cout << "image width = " << image_size_pyr_[0].width << std::endl;
  image_size_pyr_[0].height = static_cast<int>(fn["height"]);
  std::cout << "image height = " << image_size_pyr_[0].height << std::endl;
  fs.release();

  for (int level = 1; level <= pyr_levels_max_; ++level) {
    // Image size pyramid
    image_size_pyr_[level] = cv::Size((image_size_pyr_[level-1].width + 1) / 2,
                                      (image_size_pyr_[level-1].height + 1) / 2);
    // Camera intrinsics pyramid
    camera_matrix_pyr_[0].copyTo(camera_matrix_pyr_[level]);
    // Re-scale the camera intrinsics matrix (3x3)
    cv::Mat sub_camera_matrix =
      camera_matrix_pyr_[level](cv::Range(0, 2), cv::Range(0, 3));
    sub_camera_matrix = sub_camera_matrix / static_cast<double>(pow(2, level));
  }
}

CameraModel::~CameraModel() {}

bool CameraModel::Undistort(const cv::Mat& image_in, cv::Mat* image_out_ptr,
                            const int pyr_level) const {
  if (pyr_level > pyr_levels_max_) {
    std::cout << "Incorrect pyramid level" << std::endl;
    return false;
  }
  // Un-distort the key-frame image
  cv::Mat& image_out = *image_out_ptr;
  cv::undistort(image_in, image_out, camera_matrix_pyr_[pyr_level],
                camera_distortions_);
  return true;
}

cv::Point2f CameraModel::Project(const cv::Vec3f &point) const {
  std::vector<cv::Point3f> input;
  input.push_back(point);
  std::vector<cv::Point2f> output;
  cv::projectPoints(input, cv::Vec3f::all(0.0), cv::Vec3f::all(0.0),
                    camera_matrix_pyr_[0], camera_distortions_, output);
  return output[0];
}

cv::Point2d CameraModel::Project(const cv::Vec3d &point) const {
  std::vector<cv::Point3d> input;
  input.push_back(point);
  std::vector<cv::Point2d> output;
  cv::projectPoints(input, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                    camera_matrix_pyr_[0], camera_distortions_, output);
  return output[0];
}

}  // namespace planar_tracking
