#include "bruteforce_match.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iterator>
#include <algorithm>
#include <iostream>

namespace planar_tracking {

//  Brute-force based 2D picture matching
static inline void FindMaxLoc(const cv::Mat& mtx,
                              double* max_val,
                              cv::Point* max_loc) {
  double min_val;
  cv::Point min_loc;
  cv::minMaxLoc(mtx, &min_val, max_val, &min_loc, max_loc);
}

static inline bool PointInRange(const cv::Size image_size, const int xpos,
                                const int ypos, const int x_radius,
                                const int y_radius) {
  return ((xpos >= x_radius) && (ypos >= y_radius) &&
         (ypos + y_radius < image_size.height) &&
         (xpos + x_radius < image_size.width));
}

BruteForceMatch::BruteForceMatch()
  : PictureMatch(),
    homography_min_inliers_(INT_MAX),
    homography_reprojthr_(0) {}

BruteForceMatch::~BruteForceMatch() {}

void BruteForceMatch::Setup(const cv::Mat& template_image,
                            const int minimum_inliers,
                            const double ransac_reprojthr,
                            const BlockMatchingInfo& bm_info) {
  template_image.copyTo(template_image_);
  homography_min_inliers_ = minimum_inliers;
  homography_reprojthr_ = ransac_reprojthr;
  bm_info_ = bm_info;
}

double BruteForceMatch::CalcMatchingNcc(const cv::Mat& a, const cv::Mat& b) {
  double ncc = -1.0;
  if (a.rows != b.rows || a.cols != b.cols) {
    std::cout << "The size of a must be equal to b" << std::endl;
    return ncc;
  }

  cv::Mat _dist;
  cv::matchTemplate(a, b, _dist, cv::TM_CCOEFF_NORMED);
  if (_dist.type() == CV_32F) {
    ncc = _dist.at<float>(0, 0);
  } else if (_dist.type() == CV_64F) {
    ncc = _dist.at<double>(0, 0);
  } else {
    ncc = -1;
  }
  return ncc;
}

void BruteForceMatch::GenerateTssScanningScheme(
    const int step_size,
    std::vector<cv::Point2f>* _ss,
    int *ss_count) {
  _ss->resize(9);
  cv::Point2f *ss = &(*_ss)[0];
  *ss_count = 0;
  for (int m = -step_size; m <= step_size; m += step_size) {
    for (int n = -step_size; n <= step_size; n += step_size, ++(*ss_count)) {
      ss[*ss_count].y = m;
      ss[*ss_count].x = n;
    }
  }
}

void BruteForceMatch::CalcOptflowBm(const cv::Mat& prev_img, const cv::Mat& curr_img,
                                    const std::vector<cv::Point2f>& prev_points,
                                    cv::Size block_size, const int search_radius,
                                    const double accept_ncc_level,
                                    cv::OutputArray _velocity) {
  int num_prev_pts = prev_points.size();
  if (num_prev_pts <= 0) {
    std::cout << "Incorrect number of points for brute-force matching!"
              << std::endl;
    return;
  }
  _velocity.create(2, num_prev_pts, CV_32F);
  cv::Mat velocity = _velocity.getMat();

  cv::Mat map_x = cv::Mat(block_size, CV_32F);
  cv::Mat map_y = cv::Mat(block_size, CV_32F);

  const int block_width = block_size.width;
  const int block_height = block_size.height;
  int block_x_radius =
    static_cast<int>(static_cast<float>(block_width) / 2.0f);
  int block_y_radius =
    static_cast<int>(static_cast<float>(block_height) / 2.0f);
  int L = static_cast<int>(floor(log2(search_radius + 1)));
  const int step_max = static_cast<int>(pow(2, L-1));

  #pragma omp parallel for schedule (static)
  for (int i = 0; i < num_prev_pts; ++i) {
    double max_ncc = -1;
    cv::Mat ncc_mtx = cv::Mat(3, 3, CV_64F, cv::Scalar(-1));
    int x0 = static_cast<int>(prev_points[i].x + 0.5f);
    int y0 = static_cast<int>(prev_points[i].y + 0.5f);
    float dx0 = x0 - prev_points[i].x;
    float dy0 = y0 - prev_points[i].y;
    if (!PointInRange(prev_img.size(), x0, y0,
                      block_x_radius, block_y_radius)) {
      continue;
    }

    const int xs = x0 - block_x_radius;
    const int ys = y0 - block_y_radius;

    cv::Rect rect_prev(xs, ys, block_width, block_height);
    cv::Mat block_prev = prev_img(rect_prev);

    int x1 = x0;
    int y1 = y0;
    cv::Rect rect_curr(xs, ys, block_width, block_height);
    cv::Mat block_curr = curr_img(rect_curr);

    // NCC
    ncc_mtx.at<double>(1, 1) = CalcMatchingNcc(block_curr, block_prev);
    max_ncc = ncc_mtx.at<double>(1, 1);

    float* vy = reinterpret_cast<float*>(&velocity.at<float>(0, i));
    float* vx = reinterpret_cast<float*>(&velocity.at<float>(1, i));
    *vy = FLT_MAX;
    *vx = FLT_MAX;

    // Three-step search
    int step_size = step_max;
    while (step_size >= 1) {
      // Scanning scheme coordinates
      std::vector<cv::Point2f> _ss;
      int ss_count = 0;
      // Dynamic scanning scheme
      GenerateTssScanningScheme(step_size, &_ss, &ss_count);
      cv::Point2f* ss = &_ss[0];
      for (int k = 0; k < ss_count; ++k) {
        int dx = ss[k].x;
        int dy = ss[k].y;
        int x2 = x1 + dx;
        int y2 = y1 + dy;
        if (!PointInRange(curr_img.size(), x2, y2,
                          block_x_radius, block_y_radius)) {
          continue;
        }
        int cost_row = static_cast<int>(static_cast<float>(k) / 3.0f);
        int cost_col = k % 3;
        if (cost_row == 1 && cost_col == 1) {
          continue;
        }
        rect_curr.x = x2 - block_x_radius;
        rect_curr.y = y2 - block_y_radius;
        rect_curr.width = block_size.width;
        rect_curr.height = block_size.height;
        block_curr = curr_img(rect_curr);
        ncc_mtx.at<double>(cost_row, cost_col) =
          CalcMatchingNcc(block_curr, block_prev);
      }

      cv::Point max_loc;
      double tmp_dist;
      FindMaxLoc(ncc_mtx, &tmp_dist, &max_loc);
      x1 += (max_loc.x - 1) * step_size;
      y1 += (max_loc.y - 1) * step_size;
      step_size /= 2;

      ncc_mtx.at<double>(1, 1) = ncc_mtx.at<double>(max_loc.y, max_loc.x);
      max_ncc = ncc_mtx.at<double>(1, 1);
    }
    if (max_ncc >= accept_ncc_level) {
      *vy = (y1 - y0) + dy0;
      *vx = (x1 - x0) + dx0;
    }
  }
}

void BruteForceMatch::Match(const cv::Mat& image,
                            const cv::Mat& init_homography,
                            std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pair_pts,
                            cv::Mat* homography,
                            const cv::Mat& _mask) {
  cv::Mat warp_image;
  cv::warpPerspective(image, warp_image, init_homography.inv(),
                      template_image_.size());
  const cv::Size block_size = bm_info_.block_size;
  cv::Mat mask;

  std::vector<cv::Point2f> template_keypoints;
  int search_radius = bm_info_.search_radius;
  int num_GFTT_pts = bm_info_.num_features_pts;
  int min_distance = bm_info_.min_distance;
  double acceptance_ncc_level = bm_info_.accept_ncc_level;

  cv::goodFeaturesToTrack(template_image_, template_keypoints,
                          num_GFTT_pts, 0.01, min_distance, mask, 3,
                          false, 0.04);

  int num_template_pts = template_keypoints.size();
  if (num_template_pts < homography_min_inliers_) {
    return;
  }
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                            20, 0.03);
  cv::cornerSubPix(template_image_, template_keypoints, cv::Size(5, 5),
                   cv::Size(-1, -1), termcrit);
  std::vector<cv::Point2f> warp_image_keypoints;
  std::vector<cv::Point2f> template_keypoints2;
  {
    cv::Mat velocity;
    CalcOptflowBm(template_image_, warp_image, template_keypoints,
                    block_size, search_radius, acceptance_ncc_level,
                    velocity);
    // Verify the correspondence
    for (int i = 0; i < num_template_pts; ++i) {
      float dx = velocity.at<float>(1, i);
      float dy = velocity.at<float>(0, i);
      if (dx != FLT_MAX && dy != FLT_MAX) {
        float x = template_keypoints[i].x + dx;
        float y = template_keypoints[i].y + dy;
        cv::Point2f point_first(template_keypoints[i]);
        template_keypoints2.push_back(point_first);
        cv::Point2f point_second(cv::Point2f(x, y));
        warp_image_keypoints.push_back(point_second);
      }
    }
  }

  const int num_template_pts2 = template_keypoints2.size();
  if (num_template_pts2 < homography_min_inliers_) {
    return;
  }
  std::vector<cv::Point2f> image_keypoints;
  perspectiveTransform(warp_image_keypoints, image_keypoints,
                       init_homography);
  matched_pair_pts->clear();
  const int size_image_keypoints = static_cast<int>(image_keypoints.size());
  for (int i = 0; i < size_image_keypoints; ++i) {
    matched_pair_pts->push_back(std::make_pair(template_keypoints2[i],
                                               image_keypoints[i]));
  }
  *homography = findHomography(template_keypoints2, image_keypoints,
                               cv::RANSAC, 2.0);
  if (homography->at<double>(2, 2) < 0.9) {
    (*homography) = cv::Mat();
  }
}

}  // namespace planar_tracking
