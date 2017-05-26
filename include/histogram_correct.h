#ifndef HISTOGRAM_CORRECT_H_
#define HISTOGRAM_CORRECT_H_

#include "illumination_correct.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <memory>
#include <string>

namespace planar_tracking {

class HistogramCorrect : public IlluminationCorrect {
 public:
  HistogramCorrect();
  explicit HistogramCorrect(const cv::Mat& ref_image,
                            const cv::Mat& mask,
                            int nbins,
                            const double hgram_distance_thresh);
  ~HistogramCorrect();

  std::string GetName() { return "HistogramCorrect"; }
  void Correct(cv::Mat* image_ptr, const cv::Mat& mask);

  static const int NPTS = 256;
  static constexpr float kEps = 1.4901e-08;

 private:
  void EqualizeHist(cv::Mat* image_ptr);
  double CompareHist(const cv::Mat& image, const cv::Mat& mask, int method = 0);
  void ComputeImHist(const cv::Mat& image, int nbins, cv::Mat* hist_ptr,
                     const cv::Mat& mask = cv::Mat());
  void CumSum(const cv::Mat& hgram, cv::Mat* cum_ptr);
  void CreateTransformation(const cv::Mat& cum_desired, cv::Mat* hgram_actual_ptr,
                            const cv::Mat& cum_actual, const int m,
                            const int n, const int num_pixels_actual,
                            cv::Mat* t);
  void Normalize(cv::Mat* hgram_ptr, double scale_factor);
  void GrayXform(cv::Mat* image_ptr, const cv::Mat& t);

  cv::Mat mask_;
  cv::Mat hgram_template_;
  cv::Mat hgram_image_;
  int nbins_;
  double hgram_distance_thresh_;
};

}  // namespace planar_tracking

#endif  // HISTOGRAM_CORRECT_H_
