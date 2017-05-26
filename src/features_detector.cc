
#include "features_detector.h"
#include "bruteforce_match.h"
#include "camera_model.h"
#include "picture_model.h"
#include "orb_match.h"
#include "camera_frame.h"
//#include "detected_object.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <iterator>
#include <algorithm>
#include <iostream>

namespace planar_tracking {

static cv::Mat RescaleHomography(const cv::Mat& homography, double scale_x,
                                 double scale_y) {
  if (homography.empty()) {
    return cv::Mat();
  }
  cv::Mat new_homography;
  cv::Mat scale_matrix = cv::Mat::eye(3, 3, CV_64F);
  scale_matrix.at<double>(0, 0) = scale_x;
  scale_matrix.at<double>(1, 1) = scale_y;
  // The new homography
  new_homography = scale_matrix * homography * scale_matrix.inv();
  return new_homography;
}

FeaturesDetector::FeaturesDetector(const PictureDetectionOptions& options,
                                   const FeaturesDetectorOptions& fd_options)
  : PictureDetector(options), fd_options_(fd_options) {
  LoadParamFiles(fd_options_.filename_settings.c_str());
}

FeaturesDetector::~FeaturesDetector() {}

void FeaturesDetector::LoadParamFiles(const char* filename) {
  cv::FileStorage fs;
  fs.open(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Unable to open the detector config file." << std::endl;
    return;
  }
  cv::FileNode fn = fs["picture_detector"];
  cv::Mat homography_min_inliers;
  fn["homography_min_inliers"] >> homography_min_inliers;
  homography_reprojthr_ = static_cast<double>(fn["homography_reprojthr"]);
  BF_method_ = static_cast<int>(fn["BF_method"]);
  pyr_max_level_ = static_cast<int>(fn["pyr_max_level"]);
  radial_max_ = static_cast<double>(fn["radial_max"]);
  focal_scale_factor_ = static_cast<double>(fn["focal_scale_factor"]);
  pose_reproj_error_ = static_cast<double>(fn["pose_reproj_error"]);
  pose_min_inliers_ = static_cast<double>(fn["pose_min_inliers"]);
  homography_transl_thresh_ = static_cast<int>(fn["homography_transl_thresh"]);
  cv::Mat block_size;
  fn["block_size"] >> block_size;
  cv::Mat search_radius;
  fn["search_radius"] >> search_radius;
  cv::Mat num_features_pts;
  fn["num_features_pts"] >> num_features_pts;
  cv::Mat min_distance;
  fn["min_distance"] >> min_distance;
  cv::Mat accept_ncc_level;
  fn["accept_ncc_level"] >> accept_ncc_level;

  // Block matching settings
  bf_match_params_.resize(pyr_max_level_ + 1);
  homography_min_inliers_.resize(pyr_max_level_ + 1);
  for (int level = 0; level <= pyr_max_level_; ++level) {
    homography_min_inliers_[level] =
      homography_min_inliers.at<int>(level);
    int tmp_size = block_size.at<int>(level);
    bf_match_params_[level].block_size =
      cv::Size(tmp_size, tmp_size);
    bf_match_params_[level].num_features_pts =
      num_features_pts.at<int>(level);
    bf_match_params_[level].search_radius =
      search_radius.at<int>(level);
    bf_match_params_[level].min_distance =
      min_distance.at<int>(level);
    bf_match_params_[level].accept_ncc_level =
      accept_ncc_level.at<double>(level);
  }
}

bool FeaturesDetector::AddModel(std::shared_ptr<PictureModel> picture_model) {
  // set the picture model
  picture_model_ = picture_model;

  // Check the pyramid level
  if (pyr_max_level_ >= PYR_MAX_LEVEL) {
    return false;
  }

  image_size_.resize(pyr_max_level_ + 1);

  // Set the size at level 0
  image_size_[0] = picture_model_->image_.size();
  for (int i = 1; i <= pyr_max_level_; ++i) {
    image_size_[i] = cv::Size((image_size_[i-1].width + 1) / 2,
                              (image_size_[i-1].height + 1) / 2);
  }

  const cv::Mat& picture_image = picture_model_->image_;
  cv::buildPyramid(picture_image, picture_img_, pyr_max_level_);

  OrbMatch* Orb_match_initiator = new OrbMatch();
  Orb_match_initiator->Setup(picture_img_[pyr_max_level_],
                             homography_min_inliers_[pyr_max_level_],
                             homography_reprojthr_,
                             fd_options_.filename_settings.c_str());
  match_initiator_.reset(Orb_match_initiator);

  for (int level = 0; level <= pyr_max_level_; ++level) {
    BruteForceMatch* bf_match_refiner = new BruteForceMatch();
    bf_match_refiner->Setup(picture_img_[level],
                            homography_min_inliers_[level],
                            homography_reprojthr_,
                            bf_match_params_[level]);
    match_refiner_[level].reset(bf_match_refiner);
  }
  return true;
}

bool FeaturesDetector::RemoveModel() {
  picture_model_.reset();
  return true;
}

bool FeaturesDetector::Detect(const Input& input, Output* output) {
  // input
  const cv::Mat& image = input.frame->camera.image;
  // output
  DetectedObject& detected_object = output->detected_object;
  cv::Mat& homography = detected_object.homography;
  std::vector<cv::Point2f>& contour_pts = detected_object.contour_pts;
  cv::Mat& cam_T_world = detected_object.cam_T_world;
  detected_object.keyframe = cv::Mat();

  if (image.empty() || !picture_model_) {
     std::cout << "Error:frame image is empty" << std::endl;
     return false;
  }

  std::vector<cv::Mat> pyr_image;
  cv::buildPyramid(image, pyr_image, pyr_max_level_);
  std::vector<std::pair<cv::Point2f, cv::Point2f> > matched_pair_pts;
  cv::Mat homography2;
  cv::Mat init_homography_local = cv::Mat();
  if (init_homography_.empty()) {
    cv::Mat pyr_image_udist;
    options_.camera_model->Undistort(pyr_image[pyr_max_level_],
                                     &pyr_image_udist,
                                     pyr_max_level_);
    cv::Mat img_mask = cv::Mat(image_size_[pyr_max_level_].height,
                               image_size_[pyr_max_level_].width,
                               CV_8U, cv::Scalar(0));
    int xstart = static_cast<int>(
      static_cast<float>(image_size_[pyr_max_level_].width) / 4.0f);
    int ystart = 0;
    int width = static_cast<int>(
      static_cast<float>(image_size_[pyr_max_level_].width) / 2.0f);
    int height = image_size_[pyr_max_level_].height;
    cv::Rect rect = cv::Rect(xstart, ystart, width, height);
    img_mask(rect) = 255;

    match_initiator_->Match(pyr_image_udist,
                            cv::Mat(),
                            &matched_pair_pts,
                            &init_homography_,
                            img_mask);
    int num_matched_pts = matched_pair_pts.size();

    std::cout << "Initial correspondence match inliers= "
              << num_matched_pts << std::endl;

    if (num_matched_pts < homography_min_inliers_[pyr_max_level_]) {
      std::cout << "Less than " << homography_min_inliers_[pyr_max_level_]
                << std::endl;
      return false;
    }
  } else {
    init_homography_local = init_homography_;
    double scale_factor = pow(2, -pyr_max_level_);
    init_homography_ = RescaleHomography(init_homography_, scale_factor,
                                         scale_factor);
  }
  init_homography_.copyTo(homography);
  for (int level = pyr_max_level_; level >= 0; level--) {
    cv::Mat pyr_udist_image;
    options_.camera_model->Undistort(pyr_image[level], &pyr_udist_image, level);
    homography2 = cv::Mat();
    match_refiner_[level]->Match(pyr_udist_image, homography,
                                 &matched_pair_pts, &homography2);
    if (homography2.empty()) {
      init_homography_.release();
      return false;
    }

    if (level > 0) {
      double scale_factor_x = 2;
      double scale_factor_y = 2;
      homography = RescaleHomography(homography2, scale_factor_x, scale_factor_y);
    }
  }

  if (!init_homography_local.empty()) {
    double dx = init_homography_local.at<double>(0, 2) -
                homography.at<double>(0, 2);
    double dy = init_homography_.at<double>(1, 2) -
                homography.at<double>(1, 2);
    if (fabs(dx) > homography_transl_thresh_ ||
        fabs(dy) > homography_transl_thresh_) {
      init_homography_ = cv::Mat();
      return false;
    }
  }

  bool status = ComputePose(matched_pair_pts, &cam_T_world);
  if (status) {
    std::vector<cv::Point2f>& corner_pts_2d = picture_model_->corner_pts_2d_;
    if (corner_pts_2d.size() > 0) {
      perspectiveTransform(corner_pts_2d, contour_pts, homography);
    }
    // Output
    image.copyTo(detected_object.keyframe);
  } else {
    // Reset the initial homography
    init_homography_ = cv::Mat();
  }
  return status;
}

bool FeaturesDetector::ComputePose(
    const std::vector<std::pair<cv::Point2f, cv::Point2f> >& matched_pair_pts,
    cv::Mat* picture_pose) {
  cv::Mat& camera_matrix = options_.camera_model->camera_matrix_pyr_[0];

  unsigned int num_pair_pts = matched_pair_pts.size();
  std::vector<cv::Point3f> object_points(num_pair_pts);
  std::vector<cv::Point2f> image_points(num_pair_pts);

  cv::Point2d picture_origin = picture_model_->picture_origin_;
  double meters_per_pixel = picture_model_->meters_per_pixel_;
  for (unsigned int i = 0; i < num_pair_pts; ++i) {
    object_points[i].y = static_cast<float>(meters_per_pixel *
      (matched_pair_pts[i].first.x + 0.5f) - picture_origin.x);
    object_points[i].x = static_cast<float>(meters_per_pixel *
      (matched_pair_pts[i].first.y + 0.5f) - picture_origin.y);
    object_points[i].z = 0.0f;
    image_points[i].x = matched_pair_pts[i].second.x;
    image_points[i].y = matched_pair_pts[i].second.y;
  }
  cv::Mat rvec, tvec;
  std::vector<int> inliers;
  solvePnPRansac(object_points, image_points, camera_matrix, cv::Mat_<double>(),
                 rvec, tvec, false, 100, pose_reproj_error_,
                 pose_min_inliers_, inliers, CV_P3P);
  const int size_inliers = static_cast<int>(inliers.size());
  bool status = size_inliers > homography_min_inliers_[0];
  if (status) {
    if (picture_pose_.empty()) {
      picture_pose_ = cv::Mat(6, 1, CV_64F);
    }
    cv::Mat objectPose_rvec = picture_pose_(cv::Range(0, 3), cv::Range::all());
    rvec.copyTo(objectPose_rvec);
    cv::Mat objectPose_tvec = picture_pose_(cv::Range(3, 6), cv::Range::all());
    tvec.copyTo(objectPose_tvec);
    // Copy to the output pose
    picture_pose_.copyTo(*picture_pose);
  } else {
    std::cout <<"Inliers.size() = " << inliers.size()
              << " homography_min_inliers_[0] = "
              << homography_min_inliers_[0] << std::endl;
  }
  return status;
}

}  // namespace planar_tracking
