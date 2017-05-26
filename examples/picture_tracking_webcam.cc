#include "tracking_engine.h"
#include "camera_frame.h"
#include "features_detector.h"
#include "sparse_lk_tracker.h"
#include "camera_model.h"

#include <boost/circular_buffer.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <string>

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

using planar_tracking::CameraFrame;
using planar_tracking::PictureDetector;
using planar_tracking::PictureModel;
using planar_tracking::TrackingEngine;
using planar_tracking::CameraModel;
using planar_tracking::FeaturesDetector;
using planar_tracking::SparseLKTracker;
using planar_tracking::PictureDetectionOptions;
using planar_tracking::FeaturesDetectorOptions;
using planar_tracking::PictureTrackerOptions;
using planar_tracking::SparseLKTrackerOptions;

// globals
std::shared_ptr<TrackingEngine> tracking_engine;
boost::circular_buffer<std::chrono::steady_clock::time_point> frame_timestamps;
bool visualize_images = true;
std::shared_ptr<PictureModel> picture_model;
std::shared_ptr<CameraModel> camera_model;
std::string calibration_filename;
std::string engine_settings_filename;

// Process the frame for picture detection and tracking
void ProcessFrame(std::unique_ptr<CameraFrame> frame_unique,
                  cv::Mat* image_out) {
  std::shared_ptr<CameraFrame> frame(std::move(frame_unique));

  // Monocular planar tracking
  tracking_engine->TrackMonocular(frame);

  frame_timestamps.push_back(std::chrono::steady_clock::now());

  std::chrono::milliseconds time_from_earliest_frame =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          frame_timestamps.back() - frame_timestamps.front());

  double frames_per_sec = 0;
  if (frame_timestamps.size() > 1) {
    frames_per_sec = (frame_timestamps.size() - 1) /
          (time_from_earliest_frame.count() / 1000.0);
  }

  // Visualize the planar object being tracked
  if (visualize_images) {
    if (tracking_engine) {
      tracking_engine->GetOverlayImage(image_out);
    }
    // Display fps in video
    cv::putText(*image_out, std::to_string(frames_per_sec),
                cv::Point(10, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
                CV_RGB(0, 0, 255), 2, 8, false);
  }
}

int main(int argc, char** argv) {
  // reset
  calibration_filename.clear();
  engine_settings_filename.clear();

  // Parse arguments
  {
    for (int i = 1; i < argc; ++i) {
      const char* s = argv[i];
      if (strcmp(s, "-c") == 0) {
        calibration_filename = std::string(argv[++i]);
      } else if (strcmp(s, "-t") == 0) {
        engine_settings_filename = std::string(argv[++i]);
      } else {
        fprintf(stderr, "Unknown options %s\n", s);
      }
    }
    // Check camera calibration file
    if (calibration_filename.empty()) {
      fprintf(stderr, "Missing camera calibration file.");
      return -1;
    }
    // Check engine setting file
    if (engine_settings_filename.empty()) {
      fprintf(stderr, "Missing engine setting file.");
      return -1;
    }
  }
  // Set circular buffer
  frame_timestamps.rset_capacity(60);
  // Load camera calibration
  camera_model.reset(new CameraModel(calibration_filename));
  // Load the picture model
  picture_model.reset(new PictureModel(engine_settings_filename));

  {
    // Picture detector
    PictureDetectionOptions detector_options;
    FeaturesDetectorOptions fd_options;
    detector_options.camera_model = camera_model;
    fd_options.filename_settings = engine_settings_filename;
    std::unique_ptr<FeaturesDetector> detector(new FeaturesDetector(
      detector_options, fd_options));
    std::cout << "\nFeature detector generated\n" << std::endl;
    // Picture tracker
    PictureTrackerOptions tracker_options;
    SparseLKTrackerOptions lk_options;
    tracker_options.camera_model = camera_model;
    lk_options.config_filename = engine_settings_filename;
    std::unique_ptr<SparseLKTracker> tracker(new SparseLKTracker(
      tracker_options, lk_options));
    std::cout << "\nSparseLKTracker generated\n" << std::endl;
    // construct tracking engine
    TrackingEngine::Params params;
    params.camera_model = camera_model;
    tracking_engine = std::unique_ptr<TrackingEngine>(new TrackingEngine(
      params,detector.release(), tracker.release()));

    // Add a picture model for detection and tracking
    tracking_engine->AddModel(*picture_model);
    std::cout << "\nTrackingEngine generated\n" << std::endl;
  }

  int camera_id = 0;
  VideoCapture webcam_capture;

  webcam_capture.open(camera_id);
  if (!webcam_capture.isOpened()) {
    fprintf(stderr, "Could not initialize webcam (%d) capture\n", camera_id);
    return -1;
  }

  // Set VGA camera resolution
  webcam_capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  webcam_capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  namedWindow("Picture Tracking Demo", 1);
  for(int i = 0;;i++) {
    cv::Mat view, view_gray;
    if (webcam_capture.isOpened()) {
      webcam_capture >> view;
      if (view.channels() == 3) {
        cv::cvtColor(view, view_gray, CV_BGR2GRAY);
      } else {
        view.copyTo(view_gray);
        cv::cvtColor(view_gray, view, CV_GRAY2BGR);
      }
    }
    std::unique_ptr<CameraFrame> frame(new CameraFrame);
    frame->camera.image = view_gray;
    frame->camera.is_updated = true;
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    frame->camera.timestamp = now_ns.time_since_epoch();
    frame->timestamp = frame->camera.timestamp;

    // Process the camera frame for picture tracking
    ProcessFrame(std::move(frame), &view);
    imshow("Picture Tracking Demo", view);

    char key = (char)waitKey(webcam_capture.isOpened() ? 2 : 500);
    if (key == 27) { // Press ESC key to exit
      break;
    }
  }

  return 0;
}
