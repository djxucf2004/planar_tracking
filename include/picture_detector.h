#ifndef PICTURE_DETECTOR_H_
#define PICTURE_DETECTOR_H_

#include "detected_object.h"
#include <opencv2/core/core.hpp>

#include <memory>

namespace planar_tracking {

// forward declaration
class CameraModel;
class CameraFrame;
class PictureModel;
//struct DetectedObject;

struct PictureDetectionOptions {
  PictureDetectionOptions() {}
  std::shared_ptr<CameraModel> camera_model;
};

class PictureDetector {
 public:
  struct Input {
    std::shared_ptr<CameraFrame> frame;
    cv::Mat mask;
  };

  struct Output {
    DetectedObject detected_object;
  };

  explicit PictureDetector(const PictureDetectionOptions& options)
      : options_(options) {}

  virtual ~PictureDetector() {}

  virtual bool AddModel(std::shared_ptr<PictureModel> picture_model) = 0;
  virtual bool RemoveModel() = 0;

  virtual bool Detect(const Input& input, Output* output) = 0;

 protected:
  PictureDetectionOptions options_;
};

}  // namespace planar_tracking

#endif  // PICTURE_DETECTOR_H_
