// Copyright 2026 Parth
#include "src/gpu_image.h"

#include <cuda_runtime_api.h>
#include <npp.h>

#include <cstdio>

namespace npp_pipeline {

bool CudaOk(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return true;
  std::printf("  CUDA error in %s: %s\n", what, cudaGetErrorString(err));
  return false;
}

bool NppOk(NppStatus status, const char* what) {
  // Negative codes are errors, positive ones are only warnings.
  if (status >= NPP_NO_ERROR) return true;
  std::printf("  NPP error in %s: status %d\n", what, static_cast<int>(status));
  return false;
}

bool SelectDevice(int device) {
  int count = 0;
  if (!CudaOk(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return false;
  if (device >= count) {
    std::printf("error: device %d asked for but only %d found\n", device,
                count);
    return false;
  }
  if (!CudaOk(cudaSetDevice(device), "cudaSetDevice")) return false;

  cudaDeviceProp prop;
  if (!CudaOk(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProps")) {
    return false;
  }
  int runtime = 0;
  cudaRuntimeGetVersion(&runtime);
  const NppLibraryVersion* npp = nppGetLibVersion();

  std::printf("GPU            : %s (sm_%d%d, %.1f GB, %d SMs)\n", prop.name,
              prop.major, prop.minor,
              prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
              prop.multiProcessorCount);
  std::printf("CUDA runtime   : %d.%d\n", runtime / 1000,
              (runtime % 1000) / 10);
  std::printf("NPP            : %d.%d.%d\n", npp->major, npp->minor,
              npp->build);
  return true;
}

GpuImage::GpuImage(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels) {
  if (channels == 3) {
    data_ = nppiMalloc_8u_C3(width, height, &step_);
  } else {
    data_ = nppiMalloc_8u_C1(width, height, &step_);
  }
  if (data_ == nullptr) {
    std::printf("  nppiMalloc failed for %dx%dx%d\n", width, height, channels);
  }
}

GpuImage::~GpuImage() {
  if (data_ != nullptr) nppiFree(data_);
}

bool GpuImage::Upload(const HostImage& image) {
  if (image.width != width_ || image.height != height_ ||
      image.channels != channels_) {
    std::printf("  upload size mismatch\n");
    return false;
  }
  const size_t row_bytes = static_cast<size_t>(width_) * channels_;
  return CudaOk(cudaMemcpy2D(data_, step_, image.pixels.data(), row_bytes,
                             row_bytes, height_, cudaMemcpyHostToDevice),
                "upload");
}

bool GpuImage::Download(HostImage* image) const {
  image->width = width_;
  image->height = height_;
  image->channels = channels_;
  const size_t row_bytes = static_cast<size_t>(width_) * channels_;
  image->pixels.resize(row_bytes * height_);
  return CudaOk(cudaMemcpy2D(image->pixels.data(), row_bytes, data_, step_,
                             row_bytes, height_, cudaMemcpyDeviceToHost),
                "download");
}

DeviceBuffer::DeviceBuffer(size_t bytes) {
  if (!CudaOk(cudaMalloc(&data_, bytes), "cudaMalloc")) data_ = nullptr;
}

DeviceBuffer::~DeviceBuffer() {
  if (data_ != nullptr) cudaFree(data_);
}

}  // namespace npp_pipeline
