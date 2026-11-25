// Copyright 2026 Parth
// The image operations, each one a thin wrapper around NPP calls.
// All of them read a gray 8-bit GpuImage and write one of the same size.

#ifndef SRC_NPP_OPS_H_
#define SRC_NPP_OPS_H_

#include <npp.h>

#include <string>

#include "src/config.h"
#include "src/gpu_image.h"

namespace npp_pipeline {

// RGB (3 channel) -> gray using the standard luma weights.
bool RgbToGray(const GpuImage& rgb, GpuImage* gray,
               const NppStreamContext& ctx);

// Histogram on the GPU, CDF lookup table on the host, LUT applied on the GPU.
bool EqualizeHistogram(const GpuImage& src, GpuImage* dst,
                       const NppStreamContext& ctx);

bool GaussianBlur(const GpuImage& src, GpuImage* dst, int mask_size,
                  const NppStreamContext& ctx);

bool Sharpen(const GpuImage& src, GpuImage* dst, const NppStreamContext& ctx);

// |Gx| + |Gy| of the 3x3 Sobel filters, saturated to 0..255.
bool SobelMagnitude(const GpuImage& src, GpuImage* dst,
                    const NppStreamContext& ctx);

bool CannyEdges(const GpuImage& src, GpuImage* dst, int low, int high,
                const NppStreamContext& ctx);

// Rescales src into dst (dst decides the output size).
bool Resize(const GpuImage& src, GpuImage* dst, const NppStreamContext& ctx);

// Runs one op by its --ops name.
bool RunOp(const std::string& op, const Config& cfg, const GpuImage& src,
           GpuImage* dst, const NppStreamContext& ctx);

}  // namespace npp_pipeline

#endif  // SRC_NPP_OPS_H_
