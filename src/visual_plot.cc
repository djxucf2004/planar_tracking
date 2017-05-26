#include "visual_plot.h"
#include "picture_model.h"
#include "camera_model.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace planar_tracking {

void VisualPlotImage(cv::Mat* image, const cv::Matx44d& cam_T_obj,
                     const CameraModel& camera_model) {
  const cv::Vec3f origin_3d(0, 0, 0);
  cv::Point2f origin_2d = camera_model.Project(TransformPoint3d(cam_T_obj, origin_3d));

  const cv::Scalar colors[] = {
    CV_RGB(255, 0, 0),
    CV_RGB(0, 255, 0),
    CV_RGB(0, 0, 255)
  };

  const cv::Vec3f endpts_3d[] = {
    {0.05, 0, 0},
    {0, 0.05, 0},
    {0, 0, 0.05}
  };

  for (int axis = 0; axis < 3; ++axis) {
    cv::line(*image, origin_2d,
             camera_model.Project(TransformPoint3d(cam_T_obj, endpts_3d[axis])),
             colors[axis], 2.0);
  }
}

}  // namespace planar_tracking
