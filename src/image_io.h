// Copyright 2026 Parth
// Host side image loading/saving (stb) and folder helpers.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <string>
#include <vector>

namespace npp_pipeline {

// 8-bit image in host memory, rows packed with no padding.
struct HostImage {
  int width = 0;
  int height = 0;
  int channels = 0;  // 1 = gray, 3 = RGB
  std::vector<unsigned char> pixels;
};

// Loads png/jpg/bmp/pgm/ppm/tga. Gray (+alpha) becomes 1 channel, anything
// else becomes 3 channel RGB.
bool LoadImage(const std::string& path, HostImage* image);

bool SavePng(const std::string& path, const HostImage& image);

// Sorted list of the readable images directly inside dir.
std::vector<std::string> ListImages(const std::string& dir);

bool MakeDirs(const std::string& dir);

// "data/sipi/misc_4.1.01.png" -> "misc_4.1.01"
std::string FileStem(const std::string& path);

// Puts two gray images next to each other with a white gap in between.
HostImage SideBySide(const HostImage& left, const HostImage& right);

}  // namespace npp_pipeline

#endif  // SRC_IMAGE_IO_H_
