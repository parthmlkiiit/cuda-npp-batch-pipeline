// Copyright 2026 Parth
// Run settings for the batch pipeline and the command line parser.

#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#include <string>
#include <vector>

namespace npp_pipeline {

// Everything one run needs. Defaults match run.sh.
struct Config {
  std::string input_dir = "data/sipi";
  std::string output_dir = "output";
  // Applied in this order on the grayscale image.
  std::vector<std::string> ops = {"equalize", "gauss", "canny"};
  int gauss_size = 5;
  int canny_low = 60;
  int canny_high = 150;
  int max_images = 0;        // 0 = every image in input_dir
  bool save_compare = true;  // also write a before|after picture
  int compare_height = 384;  // height of the before|after picture, 0 = full
  int device = 0;
  bool show_help = false;
};

// Fills cfg from argv (flags look like --name=value). Prints the reason and
// returns false on a bad flag or value.
bool ParseArgs(int argc, char** argv, Config* cfg);

void PrintUsage(const char* program);

void PrintConfig(const Config& cfg);

}  // namespace npp_pipeline

#endif  // SRC_CONFIG_H_
