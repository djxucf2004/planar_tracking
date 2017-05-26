#ifndef TRACKING_ENGINE_H_
#define TRACKING_ENGINE_H_
#include <unistd.h>

#include "picture_detector.h"
#include "picture_tracker.h"
#include "picture_model.h"

#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <condition_variable>

namespace planar_tracking {

class CameraFrame;

class TrackingEngine {
public:
  struct Params {
    Params()
        : run_detection_periodically(true),
          detection_period(0),
          run_detection_upon_tracking_loss(true),
          debug_draw_card_axes(true) {}

    bool run_detection_periodically;
    int detection_period;
    bool run_detection_upon_tracking_loss;
    bool debug_draw_card_axes;
    // camera model
    std::shared_ptr<CameraModel> camera_model;
  };

  TrackingEngine(const Params& params, PictureDetector* detector,
                 PictureTracker* tracker);
  ~TrackingEngine();

  // Process the input monocular camera frame
  void TrackMonocular(std::shared_ptr<CameraFrame> frame);

  void AddModel(const PictureModel& picture_model);
  bool RemoveModel();
  // Visualize by overlaying
  void GetOverlayImage(cv::Mat* image_out);

private:
  enum DetectorState {
    kIdle,
    kTriggered,
    kRunning,
    kFinished
  };

  Params params_;
  std::unique_ptr<PictureDetector> detector_;
  std::unique_ptr<PictureTracker> tracker_;

  std::shared_ptr<PictureModel> picture_model_;

  std::shared_ptr<CameraFrame> latest_frame_;
  std::chrono::nanoseconds last_detection_started_at_;

  struct {
    std::mutex mutex;
    std::condition_variable trigger_condition;
    std::thread thread;
    bool thread_should_stop;

    DetectorState state;
    std::chrono::nanoseconds time_elapsed;

    PictureDetector::Input input;
    PictureDetector::Output output;
  } detection_;

  TrackedObject latest_tracked_object_;

  void GetDetectorResults();
  void UpdateTracking(std::shared_ptr<CameraFrame> frame);
  void TriggerDetection(std::shared_ptr<CameraFrame> frame);
  void DetectorThread();
  void RunDetection();
};

}  // namespace planar_tracking

#endif  // TRACKING_ENGINE_H_
