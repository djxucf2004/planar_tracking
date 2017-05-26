#ifndef BRUTE_FORCE_MATCH_H_
#define BRUTE_FORCE_MATCH_H_

#include "picture_match.h"
#include <opencv2/core/core.hpp>

#include <vector>

namespace planar_tracking {

struct BlockMatchingInfo {
  cv::Size block_size;
  int search_radius;
  int num_features_pts;
  int min_distance;
  double accept_ncc_level;
};

class BruteForceMatch: public PictureMatch {
 public:
  BruteForceMatch();
  virtual ~BruteForceMatch();

  void Setup(const cv::Mat& template_image,
             const int homography_min_inliers,
             const double homography_reprojthr,
             const BlockMatchingInfo& bm_info);

  void Match(const cv::Mat& image, const cv::Mat& init_homography,
             std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pts,
             cv::Mat* homography,
             const cv::Mat& mask = cv::Mat()) override;

private:
  double CalcMatchingNcc(const cv::Mat& a, const cv::Mat& b);
  void GenerateTssScanningScheme(const int step_size,
                                 std::vector<cv::Point2f>* _ss,
                                 int *ss_count);

  void CalcOptflowBm(const cv::Mat& prev_img,
                     const cv::Mat& curr_img,
                     const std::vector<cv::Point2f>& keypoints,
                     cv::Size block_size,
                     const int search_radius,
                     const double accept_ncc_level,
                     cv::OutputArray velocity);

  cv::Mat template_image_;
  int homography_min_inliers_;
  double homography_reprojthr_;
  BlockMatchingInfo bm_info_;
};

}  // namespace planar_tracking

#endif  // BRUTE_FORCE_MATCH_H_
