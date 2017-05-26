#include "picture_tracker.h"

namespace planar_tracking {

PictureTracker::PictureTracker(const PictureTrackerOptions& options)
  : options_(options), tracker_ready_(false) {}

PictureTracker::~PictureTracker() {}

bool PictureTracker::status() const {
  return tracker_ready_;
}

}  // namespace planar_tracking
