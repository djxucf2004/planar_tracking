
#ifndef PICTURE_MATCH_H_
#define PICTURE_MATCH_H_

#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>

namespace planar_tracking {

class PictureMatch {
 public:
  PictureMatch() {}
  virtual ~PictureMatch() {}

  virtual void Match(const cv::Mat& image,
                     const cv::Mat& init_homography,
                     std::vector<std::pair<cv::Point2f, cv::Point2f> >* matched_pts,
                     cv::Mat* homography,
                     const cv::Mat& mask = cv::Mat()) = 0;
};

}  // namespace planar_tracking

#endif  // PICTURE_MATCH_H_
