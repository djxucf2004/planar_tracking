#include "histogram_correct.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/internal.hpp>

#include <string>

namespace planar_tracking {

IlluminationCorrect::IlluminationCorrect() {}

IlluminationCorrect::
IlluminationCorrect(const cv::Mat& template_image,
                    const cv::Mat& template_mask)
: template_image_(template_image),
  template_mask_(template_mask) {
  if (template_mask.empty()) {
    int type = template_image_.type();
    cv::Size size = cv::Size(template_image_.cols,
                             template_image_.rows);
    template_mask_ = cv::Mat::ones(size, type);
  }
}

IlluminationCorrect::~IlluminationCorrect() {}

HistogramCorrect::HistogramCorrect()
  : nbins_(256),
    hgram_distance_thresh_(0.8) {}

HistogramCorrect::
HistogramCorrect(const cv::Mat& template_image,
                 const cv::Mat& template_mask,
                 int nbins,
                 const double hgram_distance_thresh)
  : IlluminationCorrect(template_image, template_mask),
    nbins_(nbins),
    hgram_distance_thresh_(hgram_distance_thresh)
{}

HistogramCorrect::~HistogramCorrect() {}

void HistogramCorrect::Correct(cv::Mat* image_ptr, const cv::Mat& mask) {
  cv::Mat& image = *image_ptr;
  // Calculate the histogram distance between the keyframe and
  // the coming frame
  double distance_hgram = CompareHist(image, mask);
  // If the distance (correlation) is less than a threshold, apply
  // illumination correction by hisgram matching
  if (distance_hgram < hgram_distance_thresh_) {
    EqualizeHist(image_ptr);
  }
}

void HistogramCorrect::
EqualizeHist(cv::Mat* image_ptr) {
  int num_pixels_template = static_cast<int>(cv::sum(hgram_template_).val[0]);
  int num_pixels_image = static_cast<int>(cv::sum(hgram_image_).val[0]);

  // Normalize histogram if the number of pixels are different
  if (num_pixels_template != num_pixels_image) {
    double scale_factor = static_cast<double>(num_pixels_image) /
                          static_cast<double>(num_pixels_template);
    // Normalize the histogram
    Normalize(&hgram_template_, scale_factor);
  }
  // Recount the number after normalize
  int m = NPTS, n = NPTS;
  cv::Mat cum_template;
  CumSum(hgram_template_, &cum_template);
  cv::Mat cum_image;
  CumSum(hgram_image_, &cum_image);

  // Compute the transformation
  cv::Mat t;
  CreateTransformation(cum_template, &hgram_image_,
                       cum_image, m, n, num_pixels_image, &t);
  // Grayscale transformation
  GrayXform(image_ptr, t);
}

double HistogramCorrect::
CompareHist(const cv::Mat& image, const cv::Mat& image_mask, int method) {
  // Combine two masks into a new one
  cv::multiply(template_mask_, image_mask, mask_);

  // Compute the histogram of input image
  cv::Mat hgram_template_norm;
  ComputeImHist(template_image_, nbins_, &hgram_template_, mask_);
  cv::normalize(hgram_template_, hgram_template_norm, 0, 1,
                cv::NORM_MINMAX, -1, cv::Mat());
  cv::Mat hgram_image_norm;
  ComputeImHist(image, nbins_, &hgram_image_, mask_);
  cv::normalize(hgram_image_, hgram_image_norm, 0, 1,
                cv::NORM_MINMAX, -1, cv::Mat());
  double dist_hgram = cv::compareHist(hgram_template_norm, hgram_image_norm,
                                      method);
  return dist_hgram;
}

void HistogramCorrect::
ComputeImHist(const cv::Mat& image, int nbins,
              cv::Mat* hist_ptr, const cv::Mat& mask) {
  // Initialize local variables
  int hist_size[] = { nbins };
  int channels[] = { 0 };
  float hranges[] = { 0, NPTS };
  int hist_dims = 1;
  int num_image = 1;
  const float* ranges[] = { hranges };
  // Compute the histogram of reference image
  cv::calcHist(&image, num_image, channels, mask, *hist_ptr,
               hist_dims, hist_size, ranges, true, false);
}

void HistogramCorrect::
CumSum(const cv::Mat& hgram, cv::Mat* cum_ptr) {
  hgram.copyTo(*cum_ptr);
  int total = static_cast<int>(hgram.total());
  for (int i = 1; i < total; ++i) {
    cum_ptr->at<float>(i) += cum_ptr->at<float>(i - 1);
  }
}

void HistogramCorrect::
GrayXform(cv::Mat* image_ptr, const cv::Mat& t) {
  int max_t_idx = t.cols - 1;
  if (max_t_idx == 255) {
    // Perfect fit, we don't need to scale the index
    for (int i = 0; i < image_ptr->rows; ++i) {
      for (int j = 0; j < image_ptr->cols; ++j) {
        uchar val = image_ptr->at<uchar>(i, j);
        image_ptr->at<uchar>(i, j) =
          static_cast<uchar>(255.0f * t.at<float>(val) + 0.5f);
      }
    }
  }
}

void HistogramCorrect::
CreateTransformation(const cv::Mat& cum_desired,
                     cv::Mat* hgram_actual_ptr,
                     const cv::Mat& cum_actual,
                     const int m,
                     const int n,
                     const int num_pixels_actual,
                     cv::Mat* t) {
  // Pre-process the actual histogram
  hgram_actual_ptr->at<float>(0) =
    std::min(hgram_actual_ptr->at<float>(0), 0.0f);
  hgram_actual_ptr->at<float>(n - 1) =
    std::min(hgram_actual_ptr->at<float>(n - 1), 0.0f);
  cv::Mat tol = cv::Mat(m, 1, CV_32F, cv::Scalar(0.5f));
  tol = tol * hgram_actual_ptr->t();

  cv::Mat err = cum_desired * cv::Mat::ones(1, n, CV_32F) -
                cv::Mat::ones(m, 1, CV_32F) * cum_actual.t() + tol;

  float thresh_value = static_cast<float>(-num_pixels_actual * kEps);
  const float new_value = static_cast<float>(num_pixels_actual);
  // Thresholding - OpenCV's counterpart doens't provide the logic we
  // need here. What applies here is all those lower than the threshold
  // are set a specified value. IPP implements this option.
  for (int i = 0; i < err.rows; ++i) {
    for (int j = 0; j < err.cols; ++j) {
      if (err.at<float>(i, j) < thresh_value) {
        err.at<float>(i, j) = new_value;
      }
    }
  }
  // Sort the matrix in a column-wise manner
  cv::Mat err_indices;
  cv::sortIdx(err, err_indices, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
  cv::Mat dummy = err_indices(cv::Range(0, 1), cv::Range::all());

  (*t) = cv::Mat(1, dummy.cols, CV_32F, cv::Scalar(0));
  const float value = static_cast<float>(m - 1.0f);
  for (int i = 0; i < t->cols; ++i) {
    t->at<float>(i) = static_cast<float>(dummy.at<int>(i)) / value;
  }
}

void HistogramCorrect::
Normalize(cv::Mat* hgram_ptr, double scale_factor) {
  for (unsigned int i = 0; i < hgram_ptr->total(); ++i) {
    float val = hgram_ptr->at<float>(i) * scale_factor;
    hgram_ptr->at<float>(i) = static_cast<float>(floor(val + 0.5));
  }
}

}  // namespace planar_tracking
