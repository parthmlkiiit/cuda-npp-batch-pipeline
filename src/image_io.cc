// Copyright 2026 Parth
#include "src/image_io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17): portable folder listing
#include <string>
#include <system_error>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace npp_pipeline {
namespace {

namespace fs = std::filesystem;

constexpr int kCompareGap = 8;

bool HasImageExtension(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
         ext == ".pgm" || ext == ".ppm" || ext == ".tga";
}

}  // namespace

bool LoadImage(const std::string& path, HostImage* image) {
  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  if (!stbi_info(path.c_str(), &width, &height, &channels_in_file)) {
    std::printf("  cannot read %s: %s\n", path.c_str(), stbi_failure_reason());
    return false;
  }
  const int wanted = channels_in_file <= 2 ? 1 : 3;
  unsigned char* data =
      stbi_load(path.c_str(), &width, &height, &channels_in_file, wanted);
  if (data == nullptr) {
    std::printf("  cannot decode %s: %s\n", path.c_str(),
                stbi_failure_reason());
    return false;
  }
  image->width = width;
  image->height = height;
  image->channels = wanted;
  const size_t bytes = static_cast<size_t>(width) * height * wanted;
  image->pixels.assign(data, data + bytes);
  stbi_image_free(data);
  return true;
}

bool SavePng(const std::string& path, const HostImage& image) {
  const int ok =
      stbi_write_png(path.c_str(), image.width, image.height, image.channels,
                     image.pixels.data(), image.width * image.channels);
  if (!ok) std::printf("  cannot write %s\n", path.c_str());
  return ok != 0;
}

std::vector<std::string> ListImages(const std::string& dir) {
  std::vector<std::string> files;
  std::error_code err;
  for (const fs::directory_entry& entry : fs::directory_iterator(dir, err)) {
    if (entry.is_regular_file() && HasImageExtension(entry.path())) {
      files.push_back(entry.path().string());
    }
  }
  if (err)
    std::printf("cannot list %s: %s\n", dir.c_str(), err.message().c_str());
  std::sort(files.begin(), files.end());
  return files;
}

bool MakeDirs(const std::string& dir) {
  std::error_code err;
  fs::create_directories(dir, err);
  if (err) {
    std::printf("cannot create %s: %s\n", dir.c_str(), err.message().c_str());
    return false;
  }
  return true;
}

std::string FileStem(const std::string& path) {
  return fs::path(path).stem().string();
}

HostImage SideBySide(const HostImage& left, const HostImage& right) {
  HostImage out;
  out.width = left.width + kCompareGap + right.width;
  out.height = std::max(left.height, right.height);
  out.channels = 1;
  out.pixels.assign(static_cast<size_t>(out.width) * out.height, 255);
  for (int y = 0; y < left.height; ++y) {
    std::copy_n(&left.pixels[static_cast<size_t>(y) * left.width], left.width,
                &out.pixels[static_cast<size_t>(y) * out.width]);
  }
  const int x0 = left.width + kCompareGap;
  for (int y = 0; y < right.height; ++y) {
    std::copy_n(&right.pixels[static_cast<size_t>(y) * right.width],
                right.width,
                &out.pixels[static_cast<size_t>(y) * out.width + x0]);
  }
  return out;
}

}  // namespace npp_pipeline
