#ifndef TRACKED_OBJECT_H_
#define TRACKED_OBJECT_H_

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace planar_tracking {

// Convert pose to Matx44d format
template <typename Type, typename Scalar>
Type GetMatx44Pose(const cv::Mat& pose_in) {
  cv::Mat R;
  cv::Mat om = pose_in(cv::Range(0, 3), cv::Range::all());
  cv::Mat t = pose_in(cv::Range(3, 6), cv::Range::all());
  Rodrigues(om, R);
  Type pose_out = cv::Matx<Scalar,4,4>::eye();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      pose_out(i, j) = R.at<Scalar>(i, j);
    }
    pose_out(i, 3) = t.at<Scalar>(i);
  }
  return pose_out;
}

struct TrackedObject {
  cv::Mat cam_T_world;
};

}  // namespace planar_tracking

#endif  // TRACKED_OBJECT_H_
