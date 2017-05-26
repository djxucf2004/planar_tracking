#ifndef ILLUMINATION_CORRECT_H_
#define ILLUMINATION_CORRECT_H_

#include <opencv2/core/core.hpp>
#include <memory>

namespace planar_tracking {

class IlluminationCorrect {
 public:
  IlluminationCorrect();
  explicit IlluminationCorrect(const cv::Mat& template_image,
                               const cv::Mat& template_mask);
  virtual ~IlluminationCorrect();

  virtual std::string GetName() = 0;

  inline cv::Mat GetTemplateMask() {
    return template_mask_;
  }

  virtual void Correct(cv::Mat* image,
                       const cv::Mat& mask = cv::Mat()) = 0;

 protected:
  cv::Mat template_image_;
  cv::Mat template_mask_;
};

typedef std::shared_ptr<IlluminationCorrect> IlluminationCorrectPtr;
typedef std::shared_ptr<IlluminationCorrect const> IlluminationCorrectConstPtr;

}  // namespace planar_tracking

#endif  // ILLUMINATION_CORRECT_H_
