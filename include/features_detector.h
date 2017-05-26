#ifndef FEATURES_DETECTOR_H_
#define FEATURES_DETECTOR_H_

#include "picture_detector.h"
#include <opencv2/features2d/features2d.hpp>

#include <memory>
#include <string>
#include <vector>

namespace planar_tracking {

class PictureModel;
class PictureMatch;
struct BlockMatchingInfo;

struct FeaturesDetectorOptions {
  std::string filename_settings;
};

class FeaturesDetector: public PictureDetector {
 public:
  FeaturesDetector(const PictureDetectionOptions& options,
                   const FeaturesDetectorOptions& fd_options);
  virtual ~FeaturesDetector();

  bool AddModel(std::shared_ptr<PictureModel> picture_model) override;
  bool RemoveModel() override;

  bool Detect(const Input& input, Output* output) override;

  static const int PYR_MAX_LEVEL = 3;

private:
  void LoadParamFiles(const char* filename_settings);
  //void GenPyrCalibMtx();
  bool ComputePose(const std::vector<std::pair
                   <cv::Point2f, cv::Point2f> >& matched_pts,
                   cv::Mat* picture_pose);

  FeaturesDetectorOptions fd_options_;
  std::shared_ptr<PictureModel> picture_model_;

  std::shared_ptr<PictureMatch> match_initiator_;
  std::shared_ptr<PictureMatch> match_refiner_[PYR_MAX_LEVEL + 1];
  double radial_max_;

  cv::Mat init_homography_;
  int BF_method_;

  std::vector<cv::Mat> picture_img_;
  std::vector<cv::Size> image_size_;
  int pyr_max_level_;

  std::vector<BlockMatchingInfo> bf_match_params_;
  cv::Mat picture_pose_;

  // Homography parameters
  std::vector<int> homography_min_inliers_;
  double homography_reprojthr_;
  // Focal length scale factor
  double focal_scale_factor_;
  // Image reprojection error threshold by picture pose
  double pose_reproj_error_;
  // Minimum number of inliers for picture pose estimate
  int pose_min_inliers_;
  // Homography translations threshold for verification
  int homography_transl_thresh_;
};

}  // namespace planar_tracking

#endif  // FEATURES_DETECTOR_H_
