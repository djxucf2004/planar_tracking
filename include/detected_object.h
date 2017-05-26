#ifndef DETECTED_OBJECT_H_
#define DETECTED_OBJECT_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace planar_tracking {

struct DetectedObject {
  DetectedObject()
      : cam_T_world(cv::Mat()),
        keyframe(cv::Mat()),
        homography(cv::Mat()) {}
  void Reset() {
    cam_T_world = cv::Mat();
    keyframe = cv::Mat();
    contour_pts.clear();
    homography = cv::Mat();
  }
  cv::Mat cam_T_world;
  cv::Mat keyframe;
  std::vector<cv::Point2f> contour_pts;
  cv::Mat homography;
};

}  // namespace planar_tracking

#endif  // DETECTED_OBJECT_H_
