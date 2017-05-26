#include "sparse_lk_tracker.h"
#include "picture_model.h"
#include "detected_object.h"
#include "keyframe.h"
#include "camera_frame.h"
#include "camera_model.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

namespace planar_tracking {

template<class Type>
static void GenerateInlierPoints(const std::vector<Type>& inputPoints,
                                 const std::vector<uchar>& point_status,
                                 std::vector<Type>* inlierPoints,
                                 std::vector<int>* inliers = 0) {
  const int numInliers = point_status.size();
  inlierPoints->clear();
  if (inliers != 0) {
    inliers->clear();
  }
  for (int i = 0; i < numInliers; ++i) {
    if (point_status[i] == 1) {
      inlierPoints->push_back(inputPoints[i]);
      if (inliers != 0) {
        inliers->push_back(i);
      }
    }
  }
}

static void UpdatePointStatus(const std::vector<int>& inliers_index,
                              const std::vector<uchar>& inliers_status,
                              std::vector<uchar>* point_status) {
  const int num_inliers = static_cast<int>(inliers_status.size());
  for (int i = 0; i < num_inliers; ++i) {
    if (inliers_status[i] == 0) {
      (*point_status)[inliers_index[i]] = 0;
    }
  }
}

static void UpdatePointStatus(const std::vector<int>& inliersIndex,
                              const std::vector<int>& inliers_pnp,
                              std::vector<uchar>* point_status) {
  const int num_inliers_index = inliersIndex.size();
  std::vector<uchar> inliers_status(num_inliers_index, 0);
  const int num_inliers_pnp = static_cast<int>(inliers_pnp.size());
  for (int i = 0; i < num_inliers_pnp; ++i) {
    inliers_status[inliers_pnp[i]] = 1;
  }
  UpdatePointStatus(inliersIndex, inliers_status, point_status);
}

SparseLKTracker::SparseLKTracker(const PictureTrackerOptions& options,
                                 const SparseLKTrackerOptions& lk_options)
  : PictureTracker(options), lk_options_(lk_options) {
  LoadConfigParams(lk_options_.config_filename.c_str());
}

SparseLKTracker::~SparseLKTracker() {}

bool SparseLKTracker::Track(const Input& input, Output* output) {
  cv::Mat& image = input.frame->camera.image;
  cv::Mat& template_pose = output->tracked_object.cam_T_world;

  if (image.empty() || !keyframe_ptr_) {
    return false;
  }
  if (template_pose.empty()) {
    template_pose = cv::Mat(6, 1, CV_64F);
  }

  cv::Mat warp_image, undist_image;
  options_.camera_model->Undistort(image, &undist_image);

  cv::Size out_size = keyframe_ptr_->template_roi_.size();
  cv::warpPerspective(undist_image, warp_image, template_warp_.inv(), out_size);
  lk_tracker_.Calculate(warp_image, &template_points_2d_[1], &lk_status_, &lk_error_);

  cv::Mat rvec, tvec, template_pose_rvec, template_pose_tvec;
  bool use_init_guess = false;
  if (!template_pose_.empty()) {
    template_pose_rvec = template_pose_(cv::Range(0, 3), cv::Range::all());
    template_pose_rvec.copyTo(rvec);
    template_pose_tvec = template_pose_(cv::Range(3, 6), cv::Range::all());
    template_pose_tvec.copyTo(tvec);
    use_init_guess = true;
  }
  std::vector<cv::Point2f> points_image_ideal(template_points_2d_[1].size());
  cv::perspectiveTransform(template_points_2d_[1], points_image_ideal,
                           template_warp_);
  bool status = ComputePoseAndHomography(points_image_ideal, &rvec, &tvec,
                                         &lk_status_, use_init_guess);

  if (!status) {
    points_image_ideal.clear();
    template_points_2d_[0].clear();
    template_points_2d_[1].clear();
    template_pose = cv::Mat();
    //keyframe_ptr_.reset();
    return false;
  }

  template_pose_rvec = template_pose(cv::Range(0, 3), cv::Range::all());
  rvec.copyTo(template_pose_rvec);
  template_pose_tvec = template_pose(cv::Range(3, 6), cv::Range::all());
  tvec.copyTo(template_pose_tvec);

  template_pose.copyTo(template_pose_);

  points_image_ideal.clear();
  template_points_2d_[1].clear();
  return true;
}

void SparseLKTracker::AddModel(std::shared_ptr<PictureModel> picture_model) {
  picture_model_ = picture_model;
}

void SparseLKTracker::RemoveModel() {
  picture_model_.reset();
}

bool SparseLKTracker::AddObject(const DetectedObject& detected_object) {
  const cv::Mat& keyframe_image = detected_object.keyframe;
  const std::vector<cv::Point2f>& template_contours = detected_object.contour_pts;
  cv::Size2d& picture_size = picture_model_->picture_size_;

  keyframe_ptr_.reset(new KeyFrame());
  bool status = keyframe_ptr_->Setup(keyframe_image, picture_size, template_contours);
  if (!status) {
    return false;
  }

  template_warp_ = keyframe_ptr_->template_warp_;
  template_warp_init_ = template_warp_.inv();

  prev_image_ = keyframe_ptr_->template_roi_;

  FeaturePatches& feature_patches = keyframe_ptr_->feature_patches_;
  const int num_points = feature_patches.FeaturesPointsCount();

  template_points_2d_[0].resize(num_points);
  template_points_2d_[1].resize(num_points);
  std::vector<cv::Point2f>* keypoints_ptr = feature_patches.FeaturesPoints();

  for (int i = 0; i < num_points; ++i) {
    template_points_2d_[0][i] = keypoints_ptr->at(i);
  }

  lk_status_.resize(num_points, 1);
  lk_error_.resize(num_points, 0);
  cv::Size win_size = cv::Size(lk_config_.optflow_win_size, lk_config_.optflow_win_size);

  const int pyramid_levels = lk_config_.optflow_pyr_level;
  const double winsize_scale = 1.73205;
  const double min_eig_thresh = 0.0005;
  const int max_count = 30;
  const double epsilon = 0.01;
  const int flags = cv::OPTFLOW_LK_GET_MIN_EIGENVALS;
  const int nbins = 256;
  const double hgram_distance_thresh = lk_config_.hgram_distance_thresh;
  bool use_illum_correct = lk_config_.illumination_correct;
  cv::Mat template_mask = keyframe_ptr_->template_mask_;

  lk_tracker_.Setup(prev_image_, template_mask, template_points_2d_[0],
                    pyramid_levels, use_illum_correct,
                    CORRECT_USE_HISTOGRAM, win_size,
                    winsize_scale, min_eig_thresh,
                    max_count, epsilon, flags,
                    nbins, hgram_distance_thresh);
  return true;
}

bool SparseLKTracker::PointInRange(
    const cv::Size image_size, const int xpos,
    const int ypos, const int x_radius,
    const int y_radius) {
  return ((xpos >= x_radius) && (ypos >= y_radius) &&
         (ypos + y_radius < image_size.height) &&
         (xpos + x_radius < image_size.width));
}

bool SparseLKTracker::LoadConfigParams(const char* filename) {
  cv::FileStorage fs;
  fs.open(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Fails to open the config file." << std::endl;
    return false;
  }

  cv::FileNode fn = fs["picture_tracker"];
  lk_config_.min_optflow_inliers =
    static_cast<int>(fn["min_optflow_inliers"]);

  lk_config_.min_pnp_inliers =
    static_cast<int>(fn["min_pnp_inliers"]);

  lk_config_.min_homography_inliers =
    static_cast<int>(fn["min_homography_inliers"]);

  lk_config_.optflow_win_size =
    static_cast<int>(fn["optflow_win_size"]);

  lk_config_.optflow_pyr_level =
    static_cast<int>(fn["optflow_pyr_level"]);

  lk_config_.inliers_ratio_thresh =
    static_cast<double>(fn["inliers_ratio_thresh"]);

  lk_config_.hgram_distance_thresh =
    static_cast<double>(fn["hgram_distance_thresh"]);

  lk_config_.homography_reproj_thresh =
    static_cast<double>(fn["homography_reproj_thresh"]);

  lk_config_.pnp_reproj_thresh =
    static_cast<float>(fn["pnp_reproj_thresh"]);

  int temp = static_cast<int>(fn["illumination_correct"]);
  lk_config_.illumination_correct = (temp != 0) ? true : false;
  fs.release();
  return true;
}

bool SparseLKTracker::ComputePoseAndHomography(
    const std::vector<cv::Point2f>& image_points_ideal,
    cv::Mat* rvec, cv::Mat* tvec, std::vector<uchar>* inliers,
    bool use_extrinsic_guess) {
  cv::Mat& camera_matrix = options_.camera_model->camera_matrix_pyr_[0];

  std::vector<int> inliers_index;
  std::vector<cv::Point2f> inliers_imagepts_ideal;
  std::vector<cv::Point2f> inliers_keyframe_pts;
  GenerateInlierPoints<cv::Point2f>(image_points_ideal, *inliers,
                                    &inliers_imagepts_ideal, &inliers_index);
  GenerateInlierPoints<cv::Point2f>(template_points_2d_[0],
                                    *inliers, &inliers_keyframe_pts);
  std::vector<uchar> inliers_homography;
  int inliers_keyframe_pts_size = static_cast<int>(inliers_keyframe_pts.size());
  if (inliers_keyframe_pts_size < lk_config_.min_optflow_inliers) {
    return false;
  }

  // Update warp transformation
  template_warp_ = cv::findHomography(inliers_keyframe_pts,
                                      inliers_imagepts_ideal,
                                      cv::RANSAC,
                                      lk_config_.homography_reproj_thresh,
                                      inliers_homography);
  if (template_warp_.at<double>(2, 2) < kHomograpyThresh) {
    return false;
  }

  int num_inliers_homography = cv::countNonZero(inliers_homography);
  if (num_inliers_homography < lk_config_.min_homography_inliers) {
    return false;
  }
  UpdatePointStatus(inliers_index, inliers_homography, inliers);

  std::vector<cv::Point3f> inliers_objectpts;
  GenerateInlierPoints<cv::Point3f>(keyframe_ptr_->template_points_3d_,
                                    *inliers, &inliers_objectpts,
                                    &inliers_index);

  std::vector<cv::Point2f> inliers_imagepts_ori;
  GenerateInlierPoints<cv::Point2f>(image_points_ideal, *inliers,
                                    &inliers_imagepts_ori);

  FeaturePatches& feature_patches = keyframe_ptr_->feature_patches_;
  double inliers_ratio = static_cast<double>(num_inliers_homography) /
    static_cast<double>(feature_patches.FeaturesPointsCount());

  if (inliers_ratio > lk_config_.inliers_ratio_thresh) {
    solvePnP(inliers_objectpts, inliers_imagepts_ori, camera_matrix,
             cv::Mat_<double>(), *rvec, *tvec, use_extrinsic_guess);
  } else {
    std::vector<int> inliers_pnp;
    const int iter_num = 100;
    solvePnPRansac(inliers_objectpts, inliers_imagepts_ori,
                   camera_matrix, cv::Mat_<double>(), *rvec, *tvec,
                   use_extrinsic_guess, iter_num,
                   lk_config_.pnp_reproj_thresh,
                   lk_config_.min_pnp_inliers, inliers_pnp);
    int num_inliers_pnp = inliers_pnp.size();
    if (num_inliers_pnp < lk_config_.min_pnp_inliers) {
      return false;
    }
    UpdatePointStatus(inliers_index, inliers_pnp, inliers);
  }
  return true;
}

}  // namespace planar_tracking
