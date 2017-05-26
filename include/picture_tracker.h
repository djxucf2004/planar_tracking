#ifndef PICTURE_TRACKER_H_
#define PICTURE_TRACKER_H_

#include "tracked_object.h"
#include <opencv2/core/core.hpp>

#include <vector>
#include <memory>

namespace planar_tracking {

// Forward declaration
class CameraFrame;
class CameraModel;
class DetectedObject;
class PictureModel;

struct PictureTrackerOptions {
  PictureTrackerOptions() {}
  std::shared_ptr<CameraModel> camera_model;
};

class PictureTracker {
 public:
  struct Input {
    std::shared_ptr<CameraFrame> frame;
  };

  struct Output {
    TrackedObject tracked_object;
  };

  explicit PictureTracker(const PictureTrackerOptions& options);
  virtual ~PictureTracker();

  virtual void AddModel(std::shared_ptr<PictureModel> picture_model) = 0;
  virtual void RemoveModel() = 0;
  virtual bool AddObject(const DetectedObject& obj) = 0;

  // object 3D pose tracking
  virtual bool Track(const Input& input, Output* output) = 0;

  // Return true if the 3D pose tracking succeeds
  bool status() const;

 protected:
  PictureTrackerOptions options_;
  // template tracking ready flag
  bool tracker_ready_;
};

}  // namespace planar_tracking

#endif  // PICTURE_TRACKER_H_
