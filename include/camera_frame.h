#ifndef CAMERA_FRAME_H_
#define CAMERA_FRAME_H_

#include <opencv2/core/core.hpp>
#include <chrono>

namespace planar_tracking {

struct CameraFrame {
  struct CameraState {
    cv::Mat image;
    // Whether this camera has sent a new image since the last capture
    bool is_updated;
    // When the image was taken
    std::chrono::nanoseconds timestamp;
  };

  CameraState camera;
  // Pose of the rig in the world frame
  cv::Matx44d world_T_cam;
  // Timestamp corresponding to the state update and to the pose estimate
  std::chrono::nanoseconds timestamp;
};

}  // namespace planar_tracking

#endif  // CAMERA_FRAME_H_
