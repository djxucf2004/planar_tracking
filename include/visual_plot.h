#ifndef VISUAL_PLOT_H_
#define VISUAL_PLOT_H_

#include <opencv2/core/core.hpp>

namespace planar_tracking {

class CameraModel;

void VisualPlotImage(cv::Mat* image, const cv::Matx44d& cam_T_obj,
                     const CameraModel& camera_model);

template<class Vec3>
Vec3 TransformPoint3d(const cv::Matx44d& pose, const Vec3& vec) {
  Vec3 result_vec;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result_vec(i) += pose(i, j) * vec(j);
    }
    result_vec(i) += pose(i, 3);
  }
  return result_vec;
}

}  // namespace planar_tracking

#endif  // VISUAL_PLOT_H_
