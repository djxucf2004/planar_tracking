#include "histogram_correct.h"
#include "mlk_pyramid.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/internal.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <float.h>
#include <stdio.h>
#include <iostream>

#define SSE2 1
#define DELTA_X_MAX 7
#define DELTA_Y_MAX 7

using std::vector;

namespace planar_tracking {

LKTrackerInvoker::LKTrackerInvoker(
    const cv::Mat& _prevImg,
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
    float _minEigThreshold) {
  prevImg = &_prevImg;
  prevDeriv = &_prevDeriv;
  nextImg = &_nextImg;
  prevPts = _prevPts;
  nextPts = _nextPts;
  status = _status;
  err = _err;
  winSize = _winSize;
  criteria = _criteria;
  level = _level;
  maxLevel = _maxLevel;
  flags = _flags;
  minEigThreshold = _minEigThreshold;
}

void LKTrackerInvoker::operator()(const cv::Range& range) const {
  const float alpha = 0.0f;
  cv::Point2f halfWin((winSize.width - 1) * 0.5f,
                      (winSize.height - 1) * 0.5f);
  const cv::Mat& I = *prevImg;
  const cv::Mat& J = *nextImg;
  const cv::Mat& derivI = *prevDeriv;

  int j, cn = I.channels(), cn2 = cn * 2;
  cv::AutoBuffer<deriv_type> _buf(winSize.area() * (cn + cn2));
  int derivDepth = cv::DataType<deriv_type>::depth;

  cv::Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
  cv::Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2),
                       (deriv_type*)_buf + winSize.area()*cn);

  for (int ptidx = range.start; ptidx < range.end; ++ptidx) {
    cv::Point2f prevPt = prevPts[ptidx]*(float)(1. / (1 << level));
    cv::Point2f nextPt;
    // Check if there is an initial predicted flow
    if (level == maxLevel) {
      if (flags & cv::OPTFLOW_USE_INITIAL_FLOW) {
        nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
      } else {
        nextPt = prevPt;
      }
    } else {
      // Update and move down to the next level
      nextPt = nextPts[ptidx] * 2.f;
    }
    // Update the initial flow before start
    nextPts[ptidx] = nextPt;

    cv::Point2i iprevPt, inextPt;
    prevPt -= halfWin;
    iprevPt.x = cvFloor(prevPt.x);
    iprevPt.y = cvFloor(prevPt.y);

    if (iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
        iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows) {
      // Out of boundaries
      if (level == 0) {
        if (status) {
          status[ptidx] = false;
        }
        if (err) {
          err[ptidx] = 0;
        }
      }
      continue;
    }

    float a = prevPt.x - iprevPt.x;
    float b = prevPt.y - iprevPt.y;
    const int W_BITS = 14, W_BITS1 = 14;
    // Bilinear interpolation : I(x1,y1), I(x1,y2), I(x2,y1), I(x2,y2)
    // a = x - x1;
    // b = y - y1
    // 1 - a = x2 - x;
    // 1 - b = y2 - y;
    const float FLT_SCALE = 1.f / (1 << 20);
    int iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
    int iw01 = cvRound(a*(1.f - b) * (1 << W_BITS));
    int iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
    int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

    int dstep = (int)(derivI.step / derivI.elemSize1());
    int stepI = (int)(I.step / I.elemSize1());
    int stepJ = (int)(J.step / J.elemSize1());
    float A11 = 0, A12 = 0, A22 = 0;
#if SSE2
    __m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
    __m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
    __m128i z = _mm_setzero_si128();
    __m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS1-1));
    __m128i qdelta = _mm_set1_epi32(1 << (W_BITS1-5-1));
    __m128 qA11 = _mm_setzero_ps();
    __m128 qA12 = _mm_setzero_ps();
    __m128 qA22 = _mm_setzero_ps();
#endif
    // Extract the patch from the first image,
    // compute covariation matrix of derivatives
    int x, y;
    for (y = 0; y < winSize.height; y++) {
      const uchar* src = (const uchar*)I.data + (y + iprevPt.y) * stepI +
                         iprevPt.x * cn;
      const deriv_type* dsrc = (const deriv_type*)derivI.data +
                               (y + iprevPt.y) * dstep + iprevPt.x * cn2;

      deriv_type* Iptr = (deriv_type*)(IWinBuf.data + y*IWinBuf.step);
      deriv_type* dIptr = (deriv_type*)(derivIWinBuf.data +
                                        y * derivIWinBuf.step);
      x = 0;
#if SSE2
      for (; x <= winSize.width * cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2) {
        __m128i v00, v01, v10, v11, t0, t1;

        v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(\
              *(const int*)(src + x)), z);
        v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(\
              *(const int*)(src + x + cn)), z);
        v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(\
              *(const int*)(src + x + stepI)), z);
        v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(\
              *(const int*)(src + x + stepI + cn)), z);

        t0 = _mm_add_epi32(_mm_madd_epi16(\
             _mm_unpacklo_epi16(v00, v01), qw0),\
             _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
        t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
             _mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0,t0));

        v00 = _mm_loadu_si128((const __m128i*)(dsrc));
        v01 = _mm_loadu_si128((const __m128i*)(dsrc + cn2));
        v10 = _mm_loadu_si128((const __m128i*)(dsrc + dstep));
        v11 = _mm_loadu_si128((const __m128i*)(dsrc + dstep + cn2));

        t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                           _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
        t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                           _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
        t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS1);
        t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS1);
        v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...

        _mm_storeu_si128((__m128i*)dIptr, v00);
        t0 = _mm_srai_epi32(v00, 16); // Iy0 Iy1 Iy2 Iy3
        t1 = _mm_srai_epi32(_mm_slli_epi32(v00, 16), 16); // Ix0 Ix1 Ix2 Ix3

        __m128 fy = _mm_cvtepi32_ps(t0);
        __m128 fx = _mm_cvtepi32_ps(t1);

        qA22 = _mm_add_ps(qA22, _mm_mul_ps(fy, fy));
        qA12 = _mm_add_ps(qA12, _mm_mul_ps(fx, fy));
        qA11 = _mm_add_ps(qA11, _mm_mul_ps(fx, fx));
      }
#endif
      for (; x < winSize.width * cn; x++, dsrc += 2, dIptr += 2) {
        int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                              src[x + stepI] * iw10 + src[x + stepI + cn]*iw11,
                              W_BITS1 - 5);
        int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                               dsrc[dstep] * iw10 + dsrc[dstep + cn2]*iw11,
                               W_BITS1);
        int iyval = CV_DESCALE(dsrc[1] * iw00 +
                               dsrc[cn2+1] * iw01 + dsrc[dstep+1]*iw10 +
                               dsrc[dstep+cn2+1]*iw11, W_BITS1);

        Iptr[x] = (short)ival;
        dIptr[0] = (short)ixval;
        dIptr[1] = (short)iyval;

        A11 += (float)(ixval * ixval);
        A12 += (float)(ixval * iyval);
        A22 += (float)(iyval * iyval);
      }
    }
#if SSE2
    float CV_DECL_ALIGNED(16) A11buf[4], A12buf[4], A22buf[4];
    _mm_store_ps(A11buf, qA11);
    _mm_store_ps(A12buf, qA12);
    _mm_store_ps(A22buf, qA22);
    A11 += A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
    A12 += A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
    A22 += A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
#endif
    A11 *= FLT_SCALE;
    A12 *= FLT_SCALE;
    A22 *= FLT_SCALE;

    // Apply the regulaized least-square method
    float _A11 = A11 + alpha;
    float _A22 = A22 + alpha;
    float D = _A11 * _A22 - A12 * A12;
    float minEig = (_A22 + _A11 - std::sqrt((_A11 - _A22) * (_A11 - _A22) +
                    4.f * A12 * A12)) / (2 * winSize.width * winSize.height);

    if (err && (flags & LKFLOW_GET_MIN_EIGENVALS) != 0) {
      err[ptidx] = (float)minEig;
    }

    if (minEig < minEigThreshold || D < FLT_EPSILON) {
      if (level == 0 && status) {
        status[ptidx] = false;
      }
      continue;
    }

    D = 1.f / D;

    nextPt -= halfWin;
    cv::Point2f prevDelta;

    for (j = 0; j < criteria.maxCount; j++) {
      inextPt.x = cvFloor(nextPt.x);
      inextPt.y = cvFloor(nextPt.y);

      if (inextPt.x < -winSize.width ||
          inextPt.x >= J.cols ||
          inextPt.y < -winSize.height ||
          inextPt.y >= J.rows) {
        if (level == 0 && status)
          status[ptidx] = false;
        break;
      }

      a = nextPt.x - inextPt.x;
      b = nextPt.y - inextPt.y;
      iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
      iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
      iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
      iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
      float b1 = 0, b2 = 0;
#if SSE2
      qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
      qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
      __m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps();
#endif
      for (y = 0; y < winSize.height; y++) {
        const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y) * stepJ +
                            inextPt.x * cn;
        const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data +
                                                     y * IWinBuf.step);
        const deriv_type* dIptr = (const deriv_type*)(derivIWinBuf.data +
                                                      y*derivIWinBuf.step);

        x = 0;
#if SSE2
        for (; x <= winSize.width * cn - 8; x += 8, dIptr += 8*2) {
          __m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x)), diff1;
          __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64(
                        (const __m128i*)(Jptr + x)), z);
          __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64(
                        (const __m128i*)(Jptr + x + cn)), z);
          __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64(
                        (const __m128i*)(Jptr + x + stepJ)), z);
          __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64(
                        (const __m128i*)(Jptr + x + stepJ + cn)), z);

          __m128i t0 = _mm_add_epi32(_mm_madd_epi16(\
                       _mm_unpacklo_epi16(v00, v01), qw0),\
                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
          __m128i t1 = _mm_add_epi32(_mm_madd_epi16(\
                       _mm_unpackhi_epi16(v00, v01), qw0),\
                       _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
          t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
          t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1-5);
          diff0 = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff0);
          diff1 = _mm_unpackhi_epi16(diff0, diff0);
          diff0 = _mm_unpacklo_epi16(diff0, diff0); // It0 It0 It1 It1...
          v00 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1...
          v01 = _mm_loadu_si128((const __m128i*)(dIptr + 8));
          v10 = _mm_mullo_epi16(v00, diff0);
          v11 = _mm_mulhi_epi16(v00, diff0);
          v00 = _mm_unpacklo_epi16(v10, v11);
          v10 = _mm_unpackhi_epi16(v10, v11);
          qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
          qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
          v10 = _mm_mullo_epi16(v01, diff1);
          v11 = _mm_mulhi_epi16(v01, diff1);
          v00 = _mm_unpacklo_epi16(v10, v11);
          v10 = _mm_unpackhi_epi16(v10, v11);
          qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
          qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
        }
#endif
        for (; x < winSize.width * cn; x++, dIptr += 2) {
          int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +\
                                Jptr[x + stepJ] * iw10 +\
                                Jptr[x + stepJ+cn] * iw11,\
                                W_BITS1-5) - Iptr[x];
          b1 += (float)(diff * dIptr[0]);
          b2 += (float)(diff * dIptr[1]);
        }
      }
#if SSE2
      float CV_DECL_ALIGNED(16) bbuf[4];
      _mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
      b1 += bbuf[0] + bbuf[2];
      b2 += bbuf[1] + bbuf[3];
#endif
      b1 *= FLT_SCALE;
      b2 *= FLT_SCALE;

      cv::Point2f delta((float)((A12 * b2 - (A22 + alpha) * b1) * D),
                        (float)((A12 * b1 - (A11 + alpha) * b2) * D));

      if (delta.x > DELTA_X_MAX || delta.y > DELTA_Y_MAX) {
        status[ptidx] = false;
        break;
      }

      nextPt += delta;
      nextPts[ptidx] = nextPt + halfWin;

      if (delta.ddot(delta) <= criteria.epsilon) {
        break;
      }

      if (j > 0 &&
          std::abs(delta.x + prevDelta.x) < 0.01 &&
          std::abs(delta.y + prevDelta.y) < 0.01) {
        nextPts[ptidx] -= delta*0.5f;
        break;
      }
      prevDelta = delta;
    }

    if (status[ptidx] &&
        err && level == 0 &&
        (flags & LKFLOW_GET_MIN_EIGENVALS) == 0) {
      cv::Point2f nextPoint = nextPts[ptidx] - halfWin;
      cv::Point inextPoint;

      inextPoint.x = cvFloor(nextPoint.x);
      inextPoint.y = cvFloor(nextPoint.y);

      if (inextPoint.x < -winSize.width ||
          inextPoint.x >= J.cols ||
          inextPoint.y < -winSize.height ||
          inextPoint.y >= J.rows) {
        if (status)
          status[ptidx] = false;
        continue;
      }

      float aa = nextPoint.x - inextPoint.x;
      float bb = nextPoint.y - inextPoint.y;
      iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
      iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
      iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
      iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
      float errval = 0.f;

      for (y = 0; y < winSize.height; y++) {
        const uchar* Jptr = (const uchar*)J.data + (y + inextPoint.y)*stepJ +
                            inextPoint.x*cn;
        const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data +
                                 y * IWinBuf.step);

        for (x = 0; x < winSize.width*cn; x++) {
          int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x+cn] * iw01 +
                                Jptr[x + stepJ] * iw10 +
                                Jptr[x + stepJ + cn] * iw11,
                                W_BITS1-5) - Iptr[x];
          errval += std::abs((float)diff);
        }
      }
      err[ptidx] = errval * 1.f / (32 * winSize.width * cn * winSize.height);
    }
  }
}

// Class LKSparseOpticalflowPyr
LKSparseOpticalflowPyr::LKSparseOpticalflowPyr()
  : pyramid_levels_(0),
    use_illum_correct_(false),
    correct_method_(CORRECT_USE_HISTOGRAM),
    winsize_(cv::Size(21, 21)),
    winsize_scale_(1.0),
    min_eig_thresh_(0.0005),
    max_count_(30),
    epsilon_(0.01),
    flags_(0),
    hgram_nbins_(256),
    hgram_distance_thresh_(0.8) {}

LKSparseOpticalflowPyr::LKSparseOpticalflowPyr(
    const cv::Mat& template_image,
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
    const double hgram_distance_thresh) {
  Setup(template_image,template_mask,
        template_fpts, pyrmid_levels,
        illumination_correct, correct_method,
        winsize, winsize_scale,
        min_eig_thresh, max_count,
        epsilon, flags,
        hgram_nbins,
        hgram_distance_thresh);
}

LKSparseOpticalflowPyr::~LKSparseOpticalflowPyr() {}

void LKSparseOpticalflowPyr::
Setup(const cv::Mat& template_image,
      const cv::Mat& template_mask,
      const std::vector<cv::Point2f>& template_fpts,
      const int pyramid_levels,
      const bool use_illum_correct,
      const int correct_method,
      const cv::Size& winsize,
      const double winsize_scale,
      const double min_eig_thresh,
      const int max_count,
      const double epsilon,
      const int flags,
      const int hgram_nbins,
      const double hgram_distance_thresh) {
  template_image_ = template_image;
  template_mask_ = template_mask;
  template_fpts_ = template_fpts;
  pyramid_levels_ = pyramid_levels;
  use_illum_correct_ = use_illum_correct;
  correct_method_ = correct_method;
  winsize_ = winsize;
  winsize_scale_ = winsize_scale;
  min_eig_thresh_ = min_eig_thresh;
  max_count_ = max_count;
  epsilon_ = epsilon;
  flags_ = flags;
  hgram_nbins_ = hgram_nbins;
  hgram_distance_thresh_ = hgram_distance_thresh;

  winsize_pyr_.resize(pyramid_levels_ + 1);
  winsize_pyr_[0] = winsize_;
  for (int level = 1; level <= pyramid_levels_; ++level) {
    int radius = cvRound(winsize_pyr_[level - 1].height/winsize_scale_)/2;
    int new_winsize = 2 * radius + 1;
    winsize_pyr_[level].width = new_winsize;
    winsize_pyr_[level].height = new_winsize;
  }
  int nlevels;
  // Template pyramid
  nlevels = BuildOpticalFlowPyramid(template_image_, &template_pyr_,
                                    winsize_pyr_,
                                    pyramid_levels_);
  if (nlevels != pyramid_levels_) {
    pyramid_levels_ = nlevels;
  }

  // Shrink the mask to avoid feature targets on the boundaries
  cv::Mat mask_erode;
  cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
  cv::erode(template_mask_, mask_erode, se);
  template_mask_ = mask_erode;
  // Template mask pyramid
  BuildOpticalFlowPyramid(template_mask_,
                          &template_mask_pyr_,
                          winsize_pyr_,
                          pyramid_levels_);

  illum_corrector_.resize(pyramid_levels_ + 1);
  for (int level = 0; level <= pyramid_levels_; level++) {
    illum_corrector_[level].reset(
        new HistogramCorrect(template_pyr_[level],
                             template_mask_pyr_[level],
                             hgram_nbins_,
                             hgram_distance_thresh_));
  }
}

void LKSparseOpticalflowPyr::
Calculate(const cv::Mat& image,
          std::vector<cv::Point2f>* next_fpts_ptr,
          std::vector<uchar>* status_ptr,
          std::vector<float>* error_ptr) {
  CalcLKSparseOpticalflowPyr(image, next_fpts_ptr, status_ptr, error_ptr);
}

void LKSparseOpticalflowPyr::
CalcImageMask(const cv::Mat& image, cv::Mat* mask_ptr) {
  // Compute the new mask
  cv::Mat& mask = *mask_ptr;
  double max_val = 1, thresh_val = 1;
  cv::threshold(image, mask, thresh_val, max_val,
                cv::THRESH_BINARY);
}

void LKSparseOpticalflowPyr::
CalcSharrDeriv(const cv::Mat& src, cv::Mat& dst) {
  // deriv_type;
  int rows = src.rows, cols = src.cols, cn = src.channels();
  int colsn = cols * cn, depth = src.depth();

  CV_Assert(depth == CV_8U);

  dst.create(rows, cols, CV_MAKETYPE(cv::DataType<deriv_type>::depth, cn*2));

  int x, y, delta = (int)cv::alignSize((cols + 2) * cn, 16);
  cv::AutoBuffer<deriv_type> _tempBuf(delta * 2 + 64);
  deriv_type *trow0 = cv::alignPtr(_tempBuf + cn, 16);
  deriv_type *trow1 = cv::alignPtr(trow0 + delta, 16);

#if SSE2
  __m128i z = _mm_setzero_si128(), c3 = _mm_set1_epi16(3);
  __m128i c10 = _mm_set1_epi16(10);
#endif

  for (y = 0; y < rows; y++) {
    const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
    const uchar* srow1 = src.ptr<uchar>(y);
    const uchar* srow2 =
      src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
    deriv_type* drow = dst.ptr<deriv_type>(y);

    // do vertical convolution
    x = 0;
#if SSE2
    for (; x <= colsn - 8; x += 8) {
      __m128i s0 =
        _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
      __m128i s1 =
        _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
      __m128i s2 =
        _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
      __m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s0, s2), c3),
                                 _mm_mullo_epi16(s1, c10));
      __m128i t1 = _mm_sub_epi16(s2, s0);
      _mm_store_si128((__m128i*)(trow0 + x), t0);
      _mm_store_si128((__m128i*)(trow1 + x), t1);
    }
#endif
    for (; x < colsn; x++) {
      int t0 = (srow0[x] + srow2[x]) * 3 + srow1[x] * 10;
      int t1 = srow2[x] - srow0[x];
      trow0[x] = (deriv_type)t0;
      trow1[x] = (deriv_type)t1;
    }

    // make border
    int x0 = (cols > 1 ? 1 : 0) * cn;
    int x1 = (cols > 1 ? cols - 2 : 0) * cn;
    for (int k = 0; k < cn; ++k) {
      trow0[-cn + k] = trow0[x0 + k];
      trow0[colsn + k] = trow0[x1 + k];
      trow1[-cn + k] = trow1[x0 + k];
      trow1[colsn + k] = trow1[x1 + k];
    }

    // do horizontal convolution, interleave
    // the results and store them to dst
    x = 0;
#if SSE2
    for (; x <= colsn - 8; x += 8) {
      __m128i s0 = _mm_loadu_si128((const __m128i*)(trow0 + x - cn));
      __m128i s1 = _mm_loadu_si128((const __m128i*)(trow0 + x + cn));
      __m128i s2 = _mm_loadu_si128((const __m128i*)(trow1 + x - cn));
      __m128i s3 = _mm_load_si128((const __m128i*)(trow1 + x));
      __m128i s4 = _mm_loadu_si128((const __m128i*)(trow1 + x + cn));

      __m128i t0 = _mm_sub_epi16(s1, s0);
      __m128i t1 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s2, s4), c3),
                                 _mm_mullo_epi16(s3, c10));
      __m128i t2 = _mm_unpacklo_epi16(t0, t1);
      t0 = _mm_unpackhi_epi16(t0, t1);
      // this can probably be replaced with aligned stores
      // if we aligned dst properly.
      _mm_storeu_si128((__m128i*)(drow + x * 2), t2);
      _mm_storeu_si128((__m128i*)(drow + x * 2 + 8), t0);
    }
#endif
    for (; x < colsn; x++) {
      deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
      deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn]) * 3 +
                                   trow1[x]*10);
      drow[x * 2] = t0; // Ix
      drow[x * 2 + 1] = t1; // Iy
    }
  }
}

int LKSparseOpticalflowPyr::BuildOpticalFlowPyramid(
    const cv::Mat& image,
    vector<cv::Mat>* pyramid_ptr,
    const std::vector<cv::Size>& win_size,
    int max_level,
    int pyr_border) {
  vector<cv::Mat>& pyramid = *pyramid_ptr;
  CV_Assert(image.depth() == CV_8U &&
            win_size[0].width > 2 &&
            win_size[0].height > 2);

  pyramid.resize(max_level + 1);
  {
    cv::Mat& temp = pyramid[0];
    if (!temp.empty()) {
      temp.adjustROI(win_size[0].height, win_size[0].height,
                     win_size[0].width, win_size[0].width);
    }
    if (temp.type() != image.type() ||
        temp.cols != win_size[0].width*2 + image.cols ||
        temp.rows != win_size[0].height*2 + image.rows) {
      temp.create(image.rows + win_size[0].height*2,
                  image.cols + win_size[0].width*2,
                  image.type());
    }
    if (pyr_border == cv::BORDER_TRANSPARENT) {
      image.copyTo(temp(cv::Rect(win_size[0].width,
                                 win_size[0].height,
                                 image.cols,
                                 image.rows)));
    } else {
      copyMakeBorder(image, temp, win_size[0].height,
                     win_size[0].height, win_size[0].width,
                     win_size[0].width, pyr_border);
    }
    temp.adjustROI(-win_size[0].height, -win_size[0].height,
                   -win_size[0].width, -win_size[0].width);
  }

  cv::Size sz = image.size();
  cv::Mat prevLevel = pyramid[0];
  cv::Mat thisLevel = prevLevel;

  for (int level = 0; level <= max_level; ++level) {
    if (level != 0) {
      cv::Mat& temp = pyramid[level];
      if (!temp.empty()) {
        temp.adjustROI(win_size[level].height, win_size[level].height,
                       win_size[level].width, win_size[level].width);
      }
      if (temp.type() != image.type() ||
          temp.cols != win_size[level].width*2 + sz.width ||
          temp.rows != win_size[level].height*2 + sz.height) {
        temp.create(sz.height + win_size[level].height * 2,
                    sz.width + win_size[level].width * 2,
                    image.type());
      }
      thisLevel = temp(cv::Rect(win_size[level].width,
                                win_size[level].height,
                                sz.width,
                                sz.height));
      pyrDown(prevLevel, thisLevel, sz);

      if (pyr_border != cv::BORDER_TRANSPARENT) {
        cv::copyMakeBorder(thisLevel,
                           temp,
                           win_size[level].height,
                           win_size[level].height,
                           win_size[level].width,
                           win_size[level].width,
                           pyr_border|cv::BORDER_ISOLATED);
      }
      temp.adjustROI(-win_size[level].height, -win_size[level].height,
                     -win_size[level].width, -win_size[level].width);
    }
    sz = cv::Size((sz.width + 1)/2,
                  (sz.height + 1)/2);

    if ((level < max_level) &&
        (sz.width <= win_size[level].width ||
         sz.height <= win_size[level].height)) {
      vector<cv::Mat>::iterator start_it = pyramid.begin() + level + 1;
      vector<cv::Mat>::iterator end_it = pyramid.end();
      pyramid.erase(start_it, end_it);
      return level;
    }
    prevLevel = thisLevel;
  }
  return max_level;
}

int LKSparseOpticalflowPyr::
CalcLKSparseOpticalflowPyr(const cv::Mat& image,
                           std::vector<cv::Point2f>* next_fpts_ptr,
                           std::vector<uchar>* status_fpts_ptr,
                           std::vector<float>* error_fpts_ptr) {
  int max_level = pyramid_levels_;
  int status = -1;
  std::vector<cv::Point2f>& next_fpts = *next_fpts_ptr;
  std::vector<uchar>& status_fpts = *status_fpts_ptr;
  std::vector<float>& error_fpts = *error_fpts_ptr;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                            max_count_, epsilon_);

  // Convert array of points into Matrix
  const int derivDepth = cv::DataType<deriv_type>::depth;

  if (max_level < 0 || winsize_.width <= 2 || winsize_.height <= 2) {
    std::cout << "Optical flows tracking::incorrect parameters." << std::endl;
    return status;
  }

  int level = 0, i, npoints;
  npoints = static_cast<int>(template_fpts_.size());
  if (npoints <= 0 ) {
    std::cout << "Optical flow tracking::no sparse features to track."
              << std::endl;
    return status;
  }

  if (!(flags_ & cv::OPTFLOW_USE_INITIAL_FLOW)) {
    // Make a copy of template feature points for
    // the initial estimate
    next_fpts = template_fpts_;
  }

  error_fpts.resize(npoints);
  status_fpts.resize(npoints);
  for (i = 0; i < npoints; i++) {
    status_fpts[i] = true;
  }

  // Image pyramid
  std::vector<cv::Mat> nextPyr, next_mask_pyr;
  BuildOpticalFlowPyramid(image, &nextPyr, winsize_pyr_, max_level);
  cv::Mat image_mask;
  CalcImageMask(image, &image_mask);
  BuildOpticalFlowPyramid(image_mask, &next_mask_pyr,
                          winsize_pyr_, max_level);

  criteria.epsilon *= criteria.epsilon;

  // dI/dx ~ Ix, dI/dy ~ Iy
  cv::Mat derivIBuf;
  derivIBuf.create(template_pyr_[0].rows + winsize_.height * 2,
                   template_pyr_[0].cols + winsize_.width * 2,
                   CV_MAKETYPE(derivDepth, template_pyr_[0].channels() * 2));


  cv::Point2f med_dxy(0.0f, 0.0f);
  for(level = max_level; level >= 0; level--) {
    cv::Mat derivI;
    cv::Size imgSize = template_pyr_[level].size();
    cv::Mat _derivI(imgSize.height + winsize_pyr_[level].height * 2,
                    imgSize.width + winsize_pyr_[level].width * 2,
                    derivIBuf.type(),
                    derivIBuf.data);
    derivI = _derivI(cv::Rect(winsize_pyr_[level].width,
                              winsize_pyr_[level].height,
                              imgSize.width,
                              imgSize.height));
    CalcSharrDeriv(template_pyr_[level], derivI);
    copyMakeBorder(derivI,
                   _derivI,
                   winsize_pyr_[level].height,
                   winsize_pyr_[level].height,
                   winsize_pyr_[level].width,
                   winsize_pyr_[level].width,
                   cv::BORDER_CONSTANT|cv::BORDER_ISOLATED);

    if (template_pyr_[level].size() != nextPyr[level].size()) {
      std::cout << "Optical flow tracking::incorrect pyramid image sizes."
                << std::endl;
      return status;
    }
    if (template_pyr_[level].type() != nextPyr[level].type()) {
      std::cout << "Optical flow tracking::incorrect pyramid image types."
                << std::endl;
      return status;
    }

    const cv::Point2f* template_fpts_ptr =\
      (const cv::Point2f*)(template_fpts_.data());
    cv::Point2f* image_fpts_ptr = (cv::Point2f*)next_fpts.data();
    uchar* status_fpts_uptr = (uchar*)status_fpts.data();
    float* error_fpts_fptr = (float*)error_fpts.data();

    // Illumination correction, if necessary.
    if (use_illum_correct_) {
      if (level != max_level) {
        int dx = round(med_dxy.x);
        int dy = round(med_dxy.y);
        if (dx != 0 || dy != 0) {
          cv::Mat temp_mask;
          cv::Mat xform = cv::Mat::zeros(2, 3, CV_64F);
          xform.at<double>(0, 0) = 1;
          xform.at<double>(1, 1) = 1;
          xform.at<double>(0, 2) = dx;
          xform.at<double>(1, 2) = dy;
          cv::Size dsize = next_mask_pyr[level].size();
          warpAffine(template_mask_pyr_[level], temp_mask,
                     xform, dsize);
          multiply(temp_mask, next_mask_pyr[level],
                   next_mask_pyr[level]);
        }
        illum_corrector_[level]->Correct(&nextPyr[level],
                                         next_mask_pyr[level]);
      }
    }

    // Apply Lucas-Kanade optical flows
    typedef LKTrackerInvoker LKTrackerInvoker;
    cv::parallel_for_(cv::Range(0, npoints),
                      LKTrackerInvoker(template_pyr_[level],
                                       derivI,
                                       nextPyr[level],
                                       template_fpts_ptr,
                                       image_fpts_ptr,
                                       status_fpts_uptr,
                                       error_fpts_fptr,
                                       winsize_pyr_[level],
                                       criteria,
                                       level,
                                       max_level,
                                       flags_,
                                       (float)min_eig_thresh_));
    med_dxy = CalcMedianOpticalFlows(npoints, template_fpts_,
                                     next_fpts, status_fpts,
                                     level, max_level);
    if (level == max_level) {
      for (int i = 0; i < npoints; ++i) {
        next_fpts[i] = template_fpts_[i]*(float)(1. / (1 << level)) + med_dxy;
      }
    }
  }
  return 0;
}

cv::Point2f LKSparseOpticalflowPyr::
CalcMedianOpticalFlows(
      int numpts,
      const std::vector<cv::Point2f>& prev_fpts,
      const std::vector<cv::Point2f>& next_fpts,
      const std::vector<uchar>& status_fpts,
      const int level,
      const int max_level) {
  std::vector<float> dx, dy;
  cv::Point2f dxy_m = cv::Point2f(0.0f, 0.0f);
  int numpts2 = 0;
  for (int k = 0; k <numpts; ++k) {
     cv::Point2f prevPt = prev_fpts[k] * (float)(1. / (1 << level));
     cv::Point2f nextPt = next_fpts[k];
     cv::Point2f dxy = cv::Point2f(nextPt.x - prevPt.x,
                                   nextPt.y - prevPt.y);
     if (status_fpts[k] &&
         fabs(dxy.x) <= DELTA_X_MAX &&
         fabs(dxy.y) <= DELTA_Y_MAX) {
       dx.push_back(dxy.x);
       dy.push_back(dxy.y);
       numpts2++;
     }
  }
  if (numpts2 > 0) {
    dxy_m.x = FindMedian(dx);
    dxy_m.y = FindMedian(dy);
  }
  return dxy_m;
}

}  // namespace planar_tracking
