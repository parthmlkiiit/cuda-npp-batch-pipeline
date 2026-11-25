// Copyright 2026 Parth
#include "src/npp_ops.h"

#include <cuda_runtime_api.h>
#include <npp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "src/config.h"
#include "src/gpu_image.h"

namespace npp_pipeline {
namespace {

constexpr int kBins = 256;

// The histogram buffer-size query takes int* in older NPP releases and
// size_t* in newer ones, so let the compiler pick the right type.
template <typename SizeT>
NppStatus HistogramBufferSize(NppStatus (*query)(NppiSize, int, SizeT*,
                                                 NppStreamContext),
                              NppiSize roi, const NppStreamContext& ctx,
                              size_t* bytes) {
  SizeT size = 0;
  const NppStatus status = query(roi, kBins + 1, &size, ctx);
  *bytes = static_cast<size_t>(size);
  return status;
}

NppiMaskSize GaussMask(int size) {
  switch (size) {
    case 3:
      return NPP_MASK_SIZE_3_X_3;
    case 7:
      return NPP_MASK_SIZE_7_X_7;
    case 9:
      return NPP_MASK_SIZE_9_X_9;
    case 11:
      return NPP_MASK_SIZE_11_X_11;
    case 13:
      return NPP_MASK_SIZE_13_X_13;
    case 15:
      return NPP_MASK_SIZE_15_X_15;
    default:
      return NPP_MASK_SIZE_5_X_5;
  }
}

// 16-bit signed scratch image for the Sobel gradients.
class Scratch16s {
 public:
  Scratch16s(int width, int height)
      : data_(nppiMalloc_16s_C1(width, height, &step_)) {}
  ~Scratch16s() {
    if (data_ != nullptr) nppiFree(data_);
  }
  Scratch16s(const Scratch16s&) = delete;
  Scratch16s& operator=(const Scratch16s&) = delete;

  bool ok() const { return data_ != nullptr; }
  Npp16s* data() const { return data_; }
  int step() const { return step_; }

 private:
  int step_ = 0;
  Npp16s* data_ = nullptr;
};

}  // namespace

bool RgbToGray(const GpuImage& rgb, GpuImage* gray,
               const NppStreamContext& ctx) {
  return NppOk(nppiRGBToGray_8u_C3C1R_Ctx(rgb.data(), rgb.step(), gray->data(),
                                          gray->step(), rgb.size(), ctx),
               "nppiRGBToGray");
}

bool EqualizeHistogram(const GpuImage& src, GpuImage* dst,
                       const NppStreamContext& ctx) {
  const NppiSize roi = src.size();

  size_t scratch_bytes = 0;
  if (!NppOk(HistogramBufferSize(nppiHistogramEvenGetBufferSize_8u_C1R_Ctx, roi,
                                 ctx, &scratch_bytes),
             "nppiHistogramEvenGetBufferSize")) {
    return false;
  }
  DeviceBuffer scratch(scratch_bytes);
  DeviceBuffer d_hist(kBins * sizeof(Npp32s));
  DeviceBuffer d_lut(kBins);
  if (!scratch.ok() || !d_hist.ok() || !d_lut.ok()) return false;

  // 257 levels -> 256 bins of width 1 covering 0..255.
  if (!NppOk(nppiHistogramEven_8u_C1R_Ctx(src.data(), src.step(), roi,
                                          d_hist.as<Npp32s>(), kBins + 1, 0,
                                          kBins, scratch.as<Npp8u>(), ctx),
             "nppiHistogramEven")) {
    return false;
  }

  Npp32s hist[kBins];
  if (!CudaOk(cudaMemcpy(hist, d_hist.as<Npp32s>(), sizeof(hist),
                         cudaMemcpyDeviceToHost),
              "histogram download")) {
    return false;
  }

  // Classic equalization: map each level through the normalized CDF.
  const int64_t total = static_cast<int64_t>(roi.width) * roi.height;
  int64_t cdf_min = 0;
  for (int i = 0; i < kBins; ++i) {
    if (hist[i] > 0) {
      cdf_min = hist[i];
      break;
    }
  }
  Npp8u lut[kBins];
  int64_t cdf = 0;
  for (int i = 0; i < kBins; ++i) {
    cdf += hist[i];
    if (total == cdf_min) {
      lut[i] = static_cast<Npp8u>(i);  // flat image, nothing to stretch
    } else {
      const double scaled =
          255.0 * static_cast<double>(cdf - cdf_min) / (total - cdf_min);
      const int level = static_cast<int>(std::lround(scaled));
      lut[i] = static_cast<Npp8u>(std::clamp(level, 0, 255));
    }
  }
  if (!CudaOk(cudaMemcpy(d_lut.as<Npp8u>(), lut, sizeof(lut),
                         cudaMemcpyHostToDevice),
              "lut upload")) {
    return false;
  }

  return NppOk(
      nppiLUTPalette_8u_C1R_Ctx(src.data(), src.step(), dst->data(),
                                dst->step(), roi, d_lut.as<Npp8u>(), 8, ctx),
      "nppiLUTPalette");
}

bool GaussianBlur(const GpuImage& src, GpuImage* dst, int mask_size,
                  const NppStreamContext& ctx) {
  return NppOk(nppiFilterGaussBorder_8u_C1R_Ctx(
                   src.data(), src.step(), src.size(), NppiPoint{0, 0},
                   dst->data(), dst->step(), src.size(), GaussMask(mask_size),
                   NPP_BORDER_REPLICATE, ctx),
               "nppiFilterGaussBorder");
}

bool Sharpen(const GpuImage& src, GpuImage* dst, const NppStreamContext& ctx) {
  return NppOk(
      nppiFilterSharpenBorder_8u_C1R_Ctx(
          src.data(), src.step(), src.size(), NppiPoint{0, 0}, dst->data(),
          dst->step(), src.size(), NPP_BORDER_REPLICATE, ctx),
      "nppiFilterSharpenBorder");
}

bool SobelMagnitude(const GpuImage& src, GpuImage* dst,
                    const NppStreamContext& ctx) {
  const NppiSize roi = src.size();
  Scratch16s gx(roi.width, roi.height);
  Scratch16s gy(roi.width, roi.height);
  if (!gx.ok() || !gy.ok()) return false;

  // 16-bit output keeps the negative gradients the 8-bit version clips.
  if (!NppOk(
          nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(
              src.data(), src.step(), roi, NppiPoint{0, 0}, gx.data(),
              gx.step(), roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx),
          "nppiFilterSobelHoriz") ||
      !NppOk(
          nppiFilterSobelVertBorder_8u16s_C1R_Ctx(
              src.data(), src.step(), roi, NppiPoint{0, 0}, gy.data(),
              gy.step(), roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx),
          "nppiFilterSobelVert")) {
    return false;
  }
  if (!NppOk(nppiAbs_16s_C1IR_Ctx(gx.data(), gx.step(), roi, ctx), "abs gx") ||
      !NppOk(nppiAbs_16s_C1IR_Ctx(gy.data(), gy.step(), roi, ctx), "abs gy") ||
      !NppOk(nppiAdd_16s_C1IRSfs_Ctx(gy.data(), gy.step(), gx.data(), gx.step(),
                                     roi, 0, ctx),
             "nppiAdd")) {
    return false;
  }
  return NppOk(nppiConvert_16s8u_C1R_Ctx(gx.data(), gx.step(), dst->data(),
                                         dst->step(), roi, ctx),
               "nppiConvert_16s8u");
}

bool CannyEdges(const GpuImage& src, GpuImage* dst, int low, int high,
                const NppStreamContext& ctx) {
  const NppiSize roi = src.size();
  int scratch_bytes = 0;
  if (!NppOk(nppiFilterCannyBorderGetBufferSize(roi, &scratch_bytes),
             "nppiFilterCannyBorderGetBufferSize")) {
    return false;
  }
  DeviceBuffer scratch(scratch_bytes);
  if (!scratch.ok()) return false;

  return NppOk(nppiFilterCannyBorder_8u_C1R_Ctx(
                   src.data(), src.step(), roi, NppiPoint{0, 0}, dst->data(),
                   dst->step(), roi, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
                   static_cast<Npp16s>(low), static_cast<Npp16s>(high),
                   nppiNormL2, NPP_BORDER_REPLICATE, scratch.as<Npp8u>(), ctx),
               "nppiFilterCannyBorder");
}

bool Resize(const GpuImage& src, GpuImage* dst, const NppStreamContext& ctx) {
  const NppiRect src_rect = {0, 0, src.width(), src.height()};
  const NppiRect dst_rect = {0, 0, dst->width(), dst->height()};
  // Super sampling averages the pixels it drops, so it only works downwards.
  const int mode =
      dst->height() < src.height() ? NPPI_INTER_SUPER : NPPI_INTER_LINEAR;
  return NppOk(nppiResize_8u_C1R_Ctx(src.data(), src.step(), src.size(),
                                     src_rect, dst->data(), dst->step(),
                                     dst->size(), dst_rect, mode, ctx),
               "nppiResize");
}

bool RunOp(const std::string& op, const Config& cfg, const GpuImage& src,
           GpuImage* dst, const NppStreamContext& ctx) {
  if (op == "equalize") return EqualizeHistogram(src, dst, ctx);
  if (op == "gauss") return GaussianBlur(src, dst, cfg.gauss_size, ctx);
  if (op == "sharpen") return Sharpen(src, dst, ctx);
  if (op == "sobel") return SobelMagnitude(src, dst, ctx);
  if (op == "canny") {
    return CannyEdges(src, dst, cfg.canny_low, cfg.canny_high, ctx);
  }
  return false;
}

}  // namespace npp_pipeline
