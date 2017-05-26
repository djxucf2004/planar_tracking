#include "keyframe.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <vector>

namespace planar_tracking {

KeyFrame::KeyFrame()
  : win_size_(cv::Size(-1, -1)) {}

KeyFrame::~KeyFrame() {}

bool KeyFrame::Setup(const cv::Mat& keyframe_image,
                     const cv::Size2d& template_size,
                     const std::vector<cv::Point2f>& template_contours,
                     const cv::Size& win_size) {
  win_size_ = win_size;
  template_contours_ = template_contours;
  template_size_ = template_size;

  double half_width = template_size_.width / 2.0;
  double half_height = template_size_.height / 2.0;

  template_corners_.resize(4);
  template_corners_[0] = cv::Point2f(-half_height, -half_width);
  template_corners_[1] = cv::Point2f(half_height, -half_width);
  template_corners_[2] = cv::Point2f(half_height, half_width);
  template_corners_[3] = cv::Point2f(-half_height, half_width);
  template_homography_ =
    cv::findHomography(template_corners_, template_contours_);
  cv::Size keyframe_size(keyframe_image.cols, keyframe_image.rows);

  template_rect_ = GetTemplateBoundingRect(keyframe_size, template_contours,
                                           win_size_);

  template_roi_ = keyframe_image(template_rect_);

  uchar frg_value = 255;
  template_mask_ = GetTemplateMask(template_roi_, template_rect_,
                                   template_contours_, frg_value);

  cv::Mat mask_erode;
  cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::erode(template_mask_, mask_erode, se);

  const int ideal_num_points = 50;
  const int bin_size = 20;
  bool status = feature_patches_.Setup(template_roi_, ideal_num_points,
                                        mask_erode, bin_size);
  if (!status) {
    return false;
  }

  int num_pts = feature_patches_.FeaturesPointsCount();

  std::vector<cv::Point2f>* keyframe_gftt_pts =
    feature_patches_.FeaturesPoints();
  std::vector<cv::Point2f> keyframe_gftt_pts_local(num_pts);
  cv::Mat H = template_homography_.inv();

  template_points_3d_.resize(num_pts);
  for (int i = 0; i < num_pts; ++i)  {
    keyframe_gftt_pts_local[i].x = (*keyframe_gftt_pts)[i].x +
      template_rect_.x;
    keyframe_gftt_pts_local[i].y = (*keyframe_gftt_pts)[i].y +
      template_rect_.y;
    double X = keyframe_gftt_pts_local[i].x*H.at<double>(0,0) +
               keyframe_gftt_pts_local[i].y*H.at<double>(0,1) + H.at<double>(0,2);
    double Y = keyframe_gftt_pts_local[i].x*H.at<double>(1,0) +
               keyframe_gftt_pts_local[i].y*H.at<double>(1,1) + H.at<double>(1,2);
    double Z = keyframe_gftt_pts_local[i].x*H.at<double>(2,0) +
               keyframe_gftt_pts_local[i].y*H.at<double>(2,1) + H.at<double>(2,2);
    Z = Z ? 1./Z : 1;
    template_points_3d_[i].x = static_cast<float>(X*Z);
    template_points_3d_[i].y = static_cast<float>(Y*Z);
    template_points_3d_[i].z = 0;
  }

  template_warp_ = cv::Mat::eye(3, 3, CV_64F);
  template_warp_.at<double>(0, 2) = template_rect_.x;
  template_warp_.at<double>(1, 2) = template_rect_.y;

  return true;
}

cv::Rect KeyFrame::GetTemplateBoundingRect(
    cv::Size img_size, const std::vector<cv::Point2f>& contour_pts,
    cv::Size win_size) {
  cv::Rect template_img_rect = cv::boundingRect(contour_pts);
  template_img_rect.x -= win_size.width;
  template_img_rect.y -= win_size.height;
  template_img_rect.width += win_size.width * 2;
  template_img_rect.height += win_size.height * 2;

  template_img_rect.x = template_img_rect.x < 0 ? 0 : template_img_rect.x;
  template_img_rect.y = template_img_rect.y < 0 ? 0 : template_img_rect.y;

  template_img_rect.width =
    (template_img_rect.x + template_img_rect.width) < img_size.width ?
    template_img_rect.width : (img_size.width - template_img_rect.x - 1);
  template_img_rect.height =
    (template_img_rect.y + template_img_rect.height) < img_size.height ?
    template_img_rect.height : (img_size.height-template_img_rect.y-1);
  return template_img_rect;
}

void KeyFrame::InitTwoChannelsComplex(cv::Mat* I, const cv::Mat& table) {
  int nRows = I->rows;
  int nCols = I->cols;
  uchar *p;
  const uchar *q;
  for (int i = 0; i < nRows; ++i) {
    p = I->ptr<uchar>(i);
    q = table.ptr<uchar>(i);
    for (int j = 0; j < nCols; ++j) {
      p[2*j] = q[j];
    }
  }
}

void KeyFrame::ExtractMask(const cv::Mat& I, cv::Mat* mask) {
  int nRows = I.rows;
  int nCols = I.cols;
  if (mask->empty()) {
    *mask = cv::Mat(nRows, nCols, CV_8U, cv::Scalar(0));
  }
  uchar *p;
  const uchar *q;
  for (int i = 0; i < nRows; ++i) {
    q = I.ptr<uchar>(i);
    p = mask->ptr<uchar>(i);
    for (int j = 0; j < nCols; ++j) {
      p[j] = q[2*j+1];
    }
  }
}

cv::Mat KeyFrame::GetTemplateMask(
    const cv::Mat& srcImg, cv::Rect template_img_rect,
    const std::vector<cv::Point2f>& contour_pts,
    uchar frgValue, uchar bkgValue) {
  std::vector<cv::Point> polygonPts;
  const int loop_iter = static_cast<int>(contour_pts.size());
  for (int i = 0; i < loop_iter; ++i) {
    int x = static_cast<int>(contour_pts[i].x - template_img_rect.x);
    int y = static_cast<int>(contour_pts[i].y - template_img_rect.y);
    x = std::max(0, x);
    x = std::min(template_img_rect.width - 1, x);
    y = std::max(0, y);
    y = std::min(template_img_rect.height - 1, y);
    polygonPts.push_back(cv::Point(x, y));
  }
  cv::Mat mask;
  cv::Mat srcMask(srcImg.rows, srcImg.cols, CV_8UC2, cv::Scalar(0, bkgValue));
  InitTwoChannelsComplex(&srcMask, srcImg);
  const cv::Point* ppt[1] = { polygonPts.data() };
  int numpts = static_cast<int>(polygonPts.size());
  int npt[] = { numpts };
  cv::fillPoly(srcMask, ppt, npt, 1, cv::Scalar(0, frgValue), 8);
  ExtractMask(srcMask, &mask);
  return mask;
}

}  // namespace planar_tracking
