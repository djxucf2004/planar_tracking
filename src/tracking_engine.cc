#include "tracking_engine.h"
#include "camera_frame.h"
#include "visual_plot.h"

#include <thread>
#include <iostream>

// #define TRACKING_ENGINE_DONT_THREAD 1

namespace planar_tracking {

TrackingEngine::TrackingEngine(const Params& params,
                               PictureDetector* detector,
                               PictureTracker* tracker)
    : params_(params), detector_(detector), tracker_(tracker) {
  detection_.state = kIdle;

#if !TRACKING_ENGINE_DONT_THREAD
  detection_.thread_should_stop = false;
  detection_.thread = std::thread(&TrackingEngine::DetectorThread, this);
#endif
}

TrackingEngine::~TrackingEngine() {
  std::unique_lock<std::mutex> detector_lock(detection_.mutex);
  detection_.thread_should_stop = true;
  detection_.trigger_condition.notify_one();
  detector_lock.unlock();

#if !TRACKING_ENGINE_DONT_THREAD
  detection_.thread.join();
#endif
}

void TrackingEngine::TrackMonocular(std::shared_ptr<CameraFrame> frame) {
  GetDetectorResults();
  UpdateTracking(frame);
  TriggerDetection(frame);
  latest_frame_ = frame;
}

void TrackingEngine::AddModel(const PictureModel& picture_model) {
  if (picture_model_ != nullptr) {
    std::cout << "The model has been loaded." << std::endl;
    return;
  }
  picture_model_ = std::make_shared<PictureModel>(picture_model);
  detector_->AddModel(picture_model_);
  tracker_->AddModel(picture_model_);
}

bool TrackingEngine::RemoveModel() {
  if (picture_model_ != nullptr) {
    detector_->RemoveModel();
    tracker_->RemoveModel();
    picture_model_.reset();
  }
  return true;
}

void TrackingEngine::GetDetectorResults() {
  std::unique_lock<std::mutex> detector_lock(detection_.mutex);
  if (detection_.state == kFinished &&
      latest_tracked_object_.cam_T_world.empty()) {
    DetectedObject& detected_object = detection_.output.detected_object;
    if (!detected_object.keyframe.empty()) {
      tracker_->AddObject(detected_object);
    }
    detection_.input.frame.reset();
    detection_.input.mask = cv::Mat();
    detection_.output.detected_object.Reset();
    detection_.state = kIdle;
  }
}

void TrackingEngine::UpdateTracking(std::shared_ptr<CameraFrame> frame) {
  PictureTracker::Input tracker_input;
  tracker_input.frame = frame;

  PictureTracker::Output tracker_output;
  tracker_->Track(tracker_input, &tracker_output);
  latest_tracked_object_ = std::move(tracker_output.tracked_object);
}

void TrackingEngine::TriggerDetection(std::shared_ptr<CameraFrame> frame) {
  std::unique_lock<std::mutex> detector_lock(detection_.mutex);
  if (detection_.state != kIdle) {
    return;
  }
  bool need_to_detect = false;
  if (!latest_frame_) {
    need_to_detect = true;
  }

  if (!need_to_detect && params_.run_detection_periodically) {
    std::chrono::steady_clock::duration duration_since_detection_started =
        frame->timestamp - last_detection_started_at_;

    std::chrono::milliseconds milliseconds_since_detection_started =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            duration_since_detection_started);

    if (milliseconds_since_detection_started.count() >=
        params_.detection_period) {
      need_to_detect = true;
    }
  }
  if (!need_to_detect) {
    return;
  }

  detection_.state = kTriggered;
  detection_.input.frame = frame;

  last_detection_started_at_ = frame->timestamp;

#if TRACKING_ENGINE_DONT_THREAD
  detection_.state = kFinished;
  RunDetection();
#else
  detection_.trigger_condition.notify_one();
#endif
}

void TrackingEngine::DetectorThread() {
  std::unique_lock<std::mutex> lock(detection_.mutex);
  if (-1 == nice(10)) {
    std::cout << "Unable to decrease detector thread's priority!" << std::endl;
  }

  while (true) {
    std::cout << "In worker thread, detector's state is "
              << detection_.state << std::endl;
    switch (detection_.state) {
      case kIdle:
        std::cout << "Detector is idle" << std::endl;
        break;
      case kTriggered:
        std::cout << "Detector is triggered!" << std::endl;
        detection_.state = kRunning;

        // Make sure the main thread can read out state while we run
        lock.unlock();
        std::cout << "Running detector" << std::endl;
        RunDetection();
        std::cout << "Finished detector" << std::endl;
        lock.lock();

        if (detection_.state != kRunning) {
        }

        detection_.state = kFinished;
        break;
      case kRunning:
        break;
      case kFinished:
        break;
    }

    if (detection_.thread_should_stop) {
      break;
    }
    detection_.trigger_condition.wait(lock);
  }
  std::cout << "Detector thread is over." << std::endl;
}

void TrackingEngine::RunDetection() {
  std::chrono::steady_clock::time_point detection_started_at =
      std::chrono::steady_clock::now();
  detector_->Detect(detection_.input, &detection_.output);

  std::chrono::steady_clock::time_point detection_ended_at =
      std::chrono::steady_clock::now();
  detection_.time_elapsed = detection_ended_at - detection_started_at;

  std::chrono::duration<double> time_elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          detection_.time_elapsed);
  std::cout<< "Detector uses " << time_elapsed_seconds.count() << "sec"
           << std::endl;
}

void TrackingEngine::GetOverlayImage(cv::Mat* image) {
  if (!latest_frame_) {
    return;
  }

  if (!latest_tracked_object_.cam_T_world.empty()) {
    cv::Matx44d cam_T_obj =
      GetMatx44Pose<cv::Matx44d, double>(latest_tracked_object_.cam_T_world);
    std::cout << "cam_T_obj= " << cam_T_obj << std::endl;
    VisualPlotImage(image, cam_T_obj, *params_.camera_model);
  }
}

}  // namespace planar_tracking
