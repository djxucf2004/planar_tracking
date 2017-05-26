#ifndef MLK_PYRAMID_H_
#define MLK_PYRAMID_H_
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <memory>

namespace planar_tracking {

#define  LKFLOW_PYR_A_READY       1
#define  LKFLOW_PYR_B_READY       2
#define  LKFLOW_INITIAL_GUESSES   4
#define  LKFLOW_GET_MIN_EIGENVALS 8

class IlluminationCorrect;
typedef std::shared_ptr<IlluminationCorrect> IlluminationCorrectPtr;

typedef short deriv_type;

struct LKTrackerInvoker : cv::ParallelLoopBody {
  LKTrackerInvoker(const cv::Mat& _prevImg,
                   const cv::Mat& _prevDeriv,
                   const cv::Mat& _nextImg,
                   const cv::Point2f* _prevPts,
                   cv::Point2f* _nextPts,
                   uchar* _status,
                   float* _err,
                   cv::Size _winSize,
                   cv::TermCriteria _criteria,
                   int _level,
                   int _maxLevel,
                   int _flags,
                   float _minEigThreshold);

  void operator()(const cv::Range& range) const;

  const cv::Mat* prevImg;
  const cv::Mat* nextImg;
  const cv::Mat* prevDeriv;
  const cv::Point2f* prevPts;
  cv::Point2f* nextPts;
  uchar* status;
  float* err;
  cv::Size winSize;
  cv::TermCriteria criteria;
  int level;
  int maxLevel;
  int flags;
  float minEigThreshold;
};

enum {
  TRANSLATIONS_XFORM = 0,
  SIMILARITY_XFORM = 1,
  AFFINE_XFORM = 2,
  HYBRID_XFORM = 3
};

enum {
  CORRECT_USE_LINEAR = 0,
  CORRECT_USE_HISTOGRAM = 1
};

template<typename _Tp>
_Tp FindMedian(const std::vector<_Tp>& src_array) {
  _Tp val = 0;
  const int nume = static_cast<int>(src_array.size());
  std::vector<_Tp> dst_array;
  cv::sort(src_array, dst_array, CV_SORT_ASCENDING);
  if (nume%2 == 0) {
    int p = (nume - 1)/2;
    val = static_cast<_Tp>((dst_array[p] + dst_array[p + 1])/2.0);
  } else {
    int p = nume / 2;
    val = dst_array[p];
  }
  return val;
}

// Class LKSparseOpticalflowPyr
class LKSparseOpticalflowPyr {
 public:
  LKSparseOpticalflowPyr();
  explicit LKSparseOpticalflowPyr(const cv::Mat& template_image,
                                  const cv::Mat& template_mask,
                                  const std::vector<cv::Point2f>& template_fpts,
                                  const int pyrmid_levels,
                                  const bool illumination_correct,
                                  const int correct_method,
                                  const cv::Size& win_size,
                                  const double win_size_scale,
                                  const double min_eig_thresh,
                                  const int max_count,
                                  const double epsilon,
                                  const int flags,
                                  const int hgram_nbins,
                                  const double hgram_distance_thresh);
  ~LKSparseOpticalflowPyr();

  void Setup(const cv::Mat& template_image,
             const cv::Mat& template_mask,
             const std::vector<cv::Point2f>& template_fpts,
             const int pyrmid_levels,
             const bool illumination_correct,
             const int correct_method,
             const cv::Size& winsize,
             const double winsize_scale,
             const double min_eig_thresh,
             const int max_count,
             const double epsilon,
             const int flags,
             const int hgram_nbins,
             const double hgram_distance_thresh);
  // Calulate the sparse optical flows
  void Calculate(const cv::Mat& image,
                 std::vector<cv::Point2f>* next_fpts_ptr,
                 std::vector<uchar>* status_ptr,
                 std::vector<float>* error_ptr);
  static void CalcImageMask(const cv::Mat& image, cv::Mat* mask_ptr);

 private:
  void CalcSharrDeriv(const cv::Mat& src, cv::Mat& dst);
  int BuildOpticalFlowPyramid(const cv::Mat& img,
                              std::vector<cv::Mat>* pyramid_ptr,
                              const std::vector<cv::Size>& win_size,
                              int maxLevel,
                              int pyrBorder = cv::BORDER_REPLICATE);
  int CalcLKSparseOpticalflowPyr(
      const cv::Mat& image,
      std::vector<cv::Point2f>* next_fpts_ptr,
      std::vector<uchar>* status_ptr,
      std::vector<float>* error_ptr);
  cv::Point2f CalcMedianOpticalFlows(
      int numpts,
      const std::vector<cv::Point2f>& prev_fpts,
      const std::vector<cv::Point2f>& next_fpts,
      const std::vector<uchar>& status_fpts,
      const int level,
      const int max_level);

 private:
  cv::Mat template_image_;
  cv::Mat template_mask_;
  std::vector<cv::Mat> template_pyr_;
  std::vector<cv::Mat> template_mask_pyr_;
  std::vector<cv::Point2f> template_fpts_;
  std::vector<IlluminationCorrectPtr> illum_corrector_;
  int pyramid_levels_;
  bool use_illum_correct_;
  int correct_method_;
  cv::Size winsize_;
  std::vector<cv::Size> winsize_pyr_;
  double winsize_scale_;
  double min_eig_thresh_;
  int max_count_;
  double epsilon_;
  int flags_;
  int hgram_nbins_;
  double hgram_distance_thresh_;
};

}  // namespace planar_tracking

#endif  // MLK_PYRAMID_H_
