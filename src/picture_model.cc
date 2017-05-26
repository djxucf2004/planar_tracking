#include <picture_model.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iostream>

namespace planar_tracking {

PictureModel::PictureModel(const std::string& filename) {
  // Load the picture model
  cv::FileStorage fs;
  fs.open(filename.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Unable to open the picture model file!" << std::endl;
    return;
  }
  cv::FileNode fn = fs["picture_model"];
  name_ = static_cast<std::string>(fn["name"]);
  std::string image_path = static_cast<std::string>(fn["image_path"]);
  image_ = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
  if (image_.empty()) {
    fprintf(stderr, "Unable to load the picture model image: %s\n", image_path.c_str());
    return;
  }
  // picture model actual size in millimeters
  picture_size_.width = static_cast<double>(fn["width"]);
  picture_size_.height = static_cast<double>(fn["height"]);
  fs.release();
  // picture image size in pixels
  image_size_ = image_.size();

  // Calibrate the planar target (millimeters per pixel)
  meters_per_pixel_ = ((static_cast<double>(picture_size_.width) /
                        static_cast<double>(image_size_.width)) +
                       (static_cast<double>(picture_size_.height) /
                        static_cast<double>(image_size_.height))) / 2.0;
  std::cout << "meters_per_pixel= " << meters_per_pixel_ << std::endl;

  // Template origin
  picture_origin_ = cv::Point2d(image_size_.width * meters_per_pixel_ / 2.0,
                                image_size_.height * meters_per_pixel_ / 2.0);
  std::cout << "x origin= " << picture_origin_.x << std::endl;
  std::cout << "y origin= " << picture_origin_.y << std::endl;
  std::cout << "template image size= " << image_size_ << "meters" << std::endl;

  // Calculate the bounding rectangle (clock-wise direction)
  corner_pts_2d_.resize(4);
  corner_pts_2d_[0] = cv::Point2f(0, 0);
  corner_pts_2d_[1] = cv::Point2f(0, image_size_.height);
  corner_pts_2d_[2] = cv::Point2f(image_size_.width, image_size_.height);
  corner_pts_2d_[3] = cv::Point2f(image_size_.width, 0);

  // Assign object corners (Z up)
  const float Y =
    static_cast<float>((image_size_.width - 1) * meters_per_pixel_ / 2.0f);
  const float X =
    static_cast<float>((image_size_.height - 1) * meters_per_pixel_ / 2.0f);
  const float Z = 0.0f;

  // Assign template 3-D position (clock-wise direction)
  corner_pts_3d_.resize(4);
  corner_pts_3d_[0] = cv::Point3f(-X, -Y, Z);
  corner_pts_3d_[1] = cv::Point3f(X, -Y, Z);
  corner_pts_3d_[2] = cv::Point3f(X, Y, Z);
  corner_pts_3d_[3] = cv::Point3f(-X, Y, Z);
}

}  // namespace planar_tracking
