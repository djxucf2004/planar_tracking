#include "orb_match.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iterator>
#include <algorithm>
#include <iostream>
#include <map>

namespace planar_tracking {

static cv::Mat FindHomography2(
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points,
    int method,
    double homography_ransac_reprojthr,
    std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pts,
    std::vector<uchar>* mask,
    int* num_inliers) {
  cv::Mat homography = findHomography(src_points, dst_points, method,
                                      homography_ransac_reprojthr, *mask);
  *num_inliers = 0;
  if (mask->size() > 0) {
    const int num_src_points = static_cast<int>(src_points.size());
    for (int k = 0; k < num_src_points; ++k) {
      if (mask->at(k)) {
        matched_pts->push_back(std::make_pair(src_points[k], dst_points[k]));
        ++(*num_inliers);
      }
    }
  }
  return homography;
}

OrbMatch::OrbMatch()
  : PictureMatch(),
    homography_min_inliers_(INT_MAX),
    homography_ransac_reprojthr_(0),
    distance_type_(cvflann::FLANN_DIST_HAMMING) {}

OrbMatch::~OrbMatch() {}

void OrbMatch::Setup(const cv::Mat& template_image,
                     const int minimum_inliers,
                     const double ransac_reprojthr,
                     const char* filename) {
  if (template_image.empty()) {
    return;
  }
  template_image.copyTo(template_image_);
  cv::FileStorage fs;
  fs.open(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Unable to load ORB matcher configuration." << std::endl;
    return;
  }

  std::string temp_str;
  cv::FileNode fn = fs["orb"];
  params_.nfeatures = static_cast<int>(fn["nfeatures"]);
  params_.scale_factor = static_cast<float>(fn["scaleFactor"]);
  params_.nlevels = static_cast<int>(fn["nlevels"]);
  params_.edge_threshold = static_cast<int>(fn["edgeThreshold"]);
  params_.first_level = static_cast<int>(fn["firstLevel"]);
  params_.wta_k = static_cast<int>(fn["WTA_K"]);
  params_.score_type = static_cast<int>(fn["scoreType"]);
  params_.patch_size = static_cast<int>(fn["patchSize"]);
  params_.fast_threshold = static_cast<int>(fn["fastThreshold"]);
  params_.strategy = (std::string)fn["strategy"];
  params_.distance_type = (std::string)fn["distanceType"];
  params_.nndr_ratio = static_cast<float>(fn["nndrRatio"]);
  params_.min_distance = static_cast<float>(fn["minDistance"]);
  params_.search_checks = static_cast<int>(fn["search_checks"]);
  params_.search_eps = static_cast<float>(fn["search_eps"]);
  temp_str = (std::string)fn["search_sorted"];
  params_.search_sorted = temp_str.compare("true") == 0 ? true : false;
  params_.lsh_key_size = static_cast<int>(fn["Lsh_key_size"]);
  params_.lsh_table_number = static_cast<int>(fn["Lsh_table_number"]);
  params_.lsh_multi_probe_level = static_cast<int>(fn["Lsh_multi_probe_level"]);
  flann_index_params_ptr_.reset(CreateFlannIndexParams(params_));
  flann_search_params_ptr_.reset(GetFlannSearchParams(params_));
  distance_type_ = GetFlannDistanceType(params_.distance_type);

  detector_ptr_.reset(CreateFeaturesDetector());
  desc_extractor_ptr_.reset(CreateDescriptorsExtractor());

  GetKeypoints(template_image_, &template_keypoints_);
  GetDescriptors(template_image_, &template_keypoints_, &template_descriptors_);
  homography_min_inliers_ = minimum_inliers;
  homography_ransac_reprojthr_ = ransac_reprojthr;
}

void OrbMatch::GetKeypoints(const cv::Mat& img,
                            std::vector<cv::KeyPoint>* keypoints,
                            const cv::Mat& mask) {
  if (detector_ptr_ != nullptr) {
    detector_ptr_->detect(img, *keypoints);
  }
}

void OrbMatch::GetDescriptors(const cv::Mat& img,
                              std::vector<cv::KeyPoint>* keypoints,
                              cv::Mat* descriptors) {
  if (desc_extractor_ptr_ != nullptr) {
    desc_extractor_ptr_->compute(img, *keypoints, *descriptors);
  }
}

cv::FeatureDetector* OrbMatch::CreateFeaturesDetector() {
  return new cv::ORB(params_.nfeatures, params_.scale_factor,
                     params_.nlevels, params_.edge_threshold,
                     params_.first_level, params_.wta_k,
                     params_.score_type, params_.patch_size);
}

cv::DescriptorExtractor* OrbMatch::CreateDescriptorsExtractor() {
  return new cv::ORB(params_.nfeatures, params_.scale_factor,
                     params_.nlevels, params_.edge_threshold,
                     params_.first_level, params_.wta_k,
                     params_.score_type, params_.patch_size);
}

cv::flann::IndexParams* OrbMatch::
CreateFlannIndexParams(const OrbMatchParams& params) {
  cv::flann::IndexParams *index_params;
  std::string str = params.strategy;
  if (str.compare("Lsh") == 0) {
    index_params = new cv::flann::LshIndexParams(params.lsh_table_number,
                                                 params.lsh_key_size,
                                                 params.lsh_multi_probe_level);
  } else {
    int table_number = 12;
    int key_size = 20;
    int multi_probe_level = 2;
    index_params = new cv::flann::LshIndexParams(table_number, key_size, multi_probe_level);
  }
  return index_params;
}

cvflann::flann_distance_t OrbMatch::
GetFlannDistanceType(const std::string& distance_type) {
  cvflann::flann_distance_t distance = cvflann::FLANN_DIST_HAMMING;

  if (distance_type.compare("MANHATTAN") == 0) {
    distance = cvflann::FLANN_DIST_MANHATTAN;
  } else if (distance_type.compare("FLANN_DIST_MINKOWSKI") == 0) {
    distance = cvflann::FLANN_DIST_MINKOWSKI;
  } else if (distance_type.compare("MAX") == 0) {
    distance = cvflann::FLANN_DIST_MAX;
  } else if (distance_type.compare("HIST_INTERSECT") == 0) {
    distance = cvflann::FLANN_DIST_HIST_INTERSECT;
  } else if (distance_type.compare("HELLINGER") == 0) {
    distance = cvflann::FLANN_DIST_HELLINGER;
  } else if (distance_type.compare("CHI_SQUARE") == 0) {
    distance = cvflann::FLANN_DIST_CHI_SQUARE;
  } else if (distance_type.compare("KULLBACK_LEIBLER") == 0) {
    distance = cvflann::FLANN_DIST_KULLBACK_LEIBLER;
  } else if (distance_type.compare("HAMMING") == 0) {
    distance = cvflann::FLANN_DIST_HAMMING;
  }
  return distance;
}

cv::flann::SearchParams* OrbMatch::
GetFlannSearchParams(const OrbMatchParams& params) {
  return new cv::flann::SearchParams(params.search_checks,
                                     params.search_eps,
                                     params.search_sorted);
}

void OrbMatch::Match(const cv::Mat& image, const cv::Mat& init_homography,
                     std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pts,
                     cv::Mat* homography, const cv::Mat& mask) {
  cv::flann::Index flann_index;
  std::vector<cv::KeyPoint> image_keypoints;
  cv::Mat image_descriptors;

  GetKeypoints(image, &image_keypoints, mask);
  int num_keypoints = static_cast<int>(image_keypoints.size());
  GetDescriptors(image, &image_keypoints, &image_descriptors);

  if (template_descriptors_.empty() ||
      num_keypoints < homography_min_inliers_) {
    return;
  }
  cv::Mat indices, dists;
  flann_index.build(image_descriptors, *flann_index_params_ptr_, distance_type_);
  indices = cv::Mat(template_descriptors_.rows, 2, CV_32SC1);
  dists = cv::Mat(template_descriptors_.rows, 2, CV_32FC1);

  flann_index.knnSearch(template_descriptors_, indices, dists, 2,
                        *flann_search_params_ptr_);
  std::map<int, int> match_index_pairs;
  for (int i = 0; i < dists.rows; ++i) {
    if (dists.at<float>(i, 0) <=
        params_.nndr_ratio * dists.at<float>(i, 1)) {
      match_index_pairs.insert(std::pair<int, int>(i, indices.at<int>(i, 0)));
    }
  }

  std::vector<cv::Point2f> mpts_template, mpts_image;
  std::map<int, int> template_pts_map;
  std::map<int, int> image_pts_map;
  std::vector<uchar> _outlier_mask;

  int j = 0;
  for (std::map<int, int>::iterator iter = match_index_pairs.begin();
       iter != match_index_pairs.end();
       ++iter) {
    cv::Point2f point1 = template_keypoints_[iter->first].pt;
    int x1 = static_cast<int>(point1.x);
    int y1 = static_cast<int>(point1.y);
    int pos1 = y1 * template_image_.cols + x1;
    cv::Point2f point2 = image_keypoints[iter->second].pt;
    int x2 = static_cast<int>(point2.x);
    int y2 = static_cast<int>(point2.y);
    int pos2 = y2 * image.cols + x2;
    if (template_pts_map.count(pos1) <= 0 && image_pts_map.count(pos2) <= 0) {
      template_pts_map[pos1] = j;
      image_pts_map[pos2] = j;
      mpts_template.push_back(point1);
      mpts_image.push_back(point2);
      ++j;
    }
  }
  const int num_mpts_template = static_cast<int>(mpts_template.size());
  if (num_mpts_template >= homography_min_inliers_) {
    int num_inliers = 0;
    cv::Mat homography_local = FindHomography2(mpts_template, mpts_image,
                                               cv::RANSAC, homography_ransac_reprojthr_,
                                               matched_pts, &_outlier_mask, &num_inliers);
    if (num_inliers >= homography_min_inliers_) {
        *homography = homography_local;
    }
  }
  flann_index.release();
  mpts_template.clear();
  mpts_image.clear();
  image_pts_map.clear();
  template_pts_map.clear();
  _outlier_mask.clear();
}

}  // namespace planar_tracking
