#ifndef SPARSE_LK_TRACKER_H_
#define SPARSE_LK_TRACKER_H_

#include "picture_tracker.h"
#include "mlk_pyramid.h"
#include <math.h>

#include <opencv2/core/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace planar_tracking {

class KeyFrame;
class PictureModel;

struct SparseLKTrackerConfig {
  SparseLKTrackerConfig()
   : min_optflow_inliers(10),
     min_pnp_inliers(15),
     min_homography_inliers(15),
     optflow_win_size(31),
     optflow_pyr_level(3),
     inliers_ratio_thresh(0.9),
     hgram_distance_thresh(0.7),
     homography_reproj_thresh(3),
     pnp_reproj_thresh(8),
     illumination_correct(true),
     min_contour_area(64) {}

  int min_optflow_inliers;
  int min_pnp_inliers;
  int min_homography_inliers;
  int optflow_win_size;
  int optflow_pyr_level;
  double inliers_ratio_thresh;
  double hgram_distance_thresh;
  double homography_reproj_thresh;
  float pnp_reproj_thresh;
  bool illumination_correct;
  double min_contour_area;
};

struct SparseLKTrackerOptions {
  std::string config_filename;
};

class SparseLKTracker : public PictureTracker {
 public:
  SparseLKTracker(const PictureTrackerOptions& options,
                  const SparseLKTrackerOptions& lk_options);

  virtual ~SparseLKTracker();

  bool Track(const Input& input, Output* output) override;

  void AddModel(std::shared_ptr<PictureModel> picture_model) override;
  void RemoveModel() override;
  bool AddObject(const DetectedObject& obj) override;

  bool PointInRange(const cv::Size image_size, const int xpos,
                    const int ypos, const int x_radius,
                    const int y_radius);

private:
  bool LoadConfigParams(const char* filename);

  bool ComputePoseAndHomography(
    const std::vector<cv::Point2f>& image_points_ideal,
    cv::Mat* rvec, cv::Mat* tvec, std::vector<uchar>* inliers,
    bool use_extrinsic_guess = false);

  SparseLKTrackerOptions lk_options_;
  // Picture model
  std::shared_ptr<PictureModel> picture_model_;
  // Keyframe
  std::shared_ptr<KeyFrame> keyframe_ptr_;

  cv::Mat prev_image_;
  cv::Mat template_pose_;

  cv::Mat template_warp_;
  cv::Mat template_warp_init_;
  std::vector<uchar> lk_status_;
  std::vector<float> lk_error_;

  std::vector<cv::Point2f> template_points_2d_[2];

  // Lukas-Kanade tracking configuraton parameters
  SparseLKTrackerConfig lk_config_;
  LKSparseOpticalflowPyr lk_tracker_;
};

}  // namespace planar_tracking

#endif  // SPARSE_LK_TRACKER_H_
