// Copyright 2026 Parth
// GPU memory helpers: a pitched 8-bit image, a plain byte buffer and error
// checks for CUDA / NPP calls.

#ifndef SRC_GPU_IMAGE_H_
#define SRC_GPU_IMAGE_H_

#include <cuda_runtime_api.h>
#include <npp.h>

#include <cstddef>

#include "src/image_io.h"

namespace npp_pipeline {

// Print the failing call and return false so callers can bail out.
bool CudaOk(cudaError_t err, const char* what);
bool NppOk(NppStatus status, const char* what);

// Selects the device and prints its name, memory and the CUDA/NPP versions.
bool SelectDevice(int device);

// 8-bit image (1 or 3 channels) in pitched device memory from nppiMalloc.
class GpuImage {
 public:
  GpuImage(int width, int height, int channels);
  ~GpuImage();

  GpuImage(const GpuImage&) = delete;
  GpuImage& operator=(const GpuImage&) = delete;

  bool ok() const { return data_ != nullptr; }
  Npp8u* data() const { return data_; }
  int step() const { return step_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  NppiSize size() const { return NppiSize{width_, height_}; }

  // Copies a host image of the same size and channel count in / out.
  bool Upload(const HostImage& image);
  bool Download(HostImage* image) const;

 private:
  int width_;
  int height_;
  int channels_;
  int step_ = 0;
  Npp8u* data_ = nullptr;
};

// Raw cudaMalloc'd scratch memory, freed when it goes out of scope.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool ok() const { return data_ != nullptr; }
  template <typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
};

}  // namespace npp_pipeline

#endif  // SRC_GPU_IMAGE_H_
