#ifndef ORB_MATCH_H_
#define ORB_MATCH_H_

#include "picture_match.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <memory>
#include <utility>

namespace planar_tracking {

class OrbMatch: public PictureMatch {
 public:
  OrbMatch();
  virtual ~OrbMatch();
  void Setup(const cv::Mat& template_image, const int minimum_inliers,
             const double ransac_reprojthr, const char* filename);

  void Match(const cv::Mat& image, const cv::Mat& init_homography,
             std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pts,
             cv::Mat* homography, const cv::Mat& mask = cv::Mat()) override;

 protected:
  struct OrbMatchParams {
    OrbMatchParams() {
      nfeatures = 500;
      scale_factor = 1.2f;
      nlevels = 8;
      edge_threshold = 31;
      first_level = 0;
      wta_k = 2;
      score_type = 0;
      patch_size = 31;
      fast_threshold = 20;

      strategy = "Lsh";
      distance_type = "HAMMING";
      nndr_ratio = 0.4;
      min_distance = 1.6;
      search_checks = 32;
      search_eps = 0;
      search_sorted = true;
      lsh_key_size = 20;
      lsh_table_number = 12;
      lsh_multi_probe_level = 2;
    }

    int nfeatures;
    float scale_factor;
    int nlevels;
    int edge_threshold;
    int first_level;
    int wta_k;
    int score_type;
    int patch_size;
    int fast_threshold;
    std::string strategy;
    std::string distance_type;
    float nndr_ratio;
    float min_distance;
    int search_checks;
    float search_eps;
    bool search_sorted;
    int lsh_table_number;
    int lsh_key_size;
    int lsh_multi_probe_level;
  };

  void GetKeypoints(const cv::Mat& img, std::vector<cv::KeyPoint>* keypoints,
                    const cv::Mat& mask = cv::Mat());
  void GetDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>* keypoints,
                      cv::Mat* descriptors);
  cv::FeatureDetector* CreateFeaturesDetector();
  cv::DescriptorExtractor* CreateDescriptorsExtractor();
  cv::flann::IndexParams* CreateFlannIndexParams(const OrbMatchParams& params);
  cvflann::flann_distance_t GetFlannDistanceType(const std::string& distance_type);
  cv::flann::SearchParams* GetFlannSearchParams(const OrbMatchParams& params);

 private:
  OrbMatchParams params_;
  std::vector<cv::KeyPoint> template_keypoints_;
  cv::Mat template_descriptors_;

  std::shared_ptr<cv::FeatureDetector> detector_ptr_;
  std::shared_ptr<cv::DescriptorExtractor> desc_extractor_ptr_;

  cv::Mat template_image_;
  int homography_min_inliers_;
  double homography_ransac_reprojthr_;

  std::shared_ptr<cv::flann::IndexParams> flann_index_params_ptr_;
  cvflann::flann_distance_t distance_type_;
  std::shared_ptr<cv::flann::SearchParams> flann_search_params_ptr_;
};

}  // namespace planar_tracking

#endif  // ORB_MATCH_H_
