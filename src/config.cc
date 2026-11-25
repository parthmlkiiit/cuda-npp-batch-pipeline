// Copyright 2026 Parth
#include "src/config.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace npp_pipeline {
namespace {

const char* const kValidOps[] = {
    "equalize", "gauss", "sharpen", "sobel", "canny",
};

bool IsValidOp(const std::string& op) {
  for (const char* valid : kValidOps) {
    if (op == valid) return true;
  }
  return false;
}

std::vector<std::string> SplitCommas(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
}

// Rejects things like "5x" or "" instead of silently reading 5 or 0.
bool ParseInt(const std::string& text, int* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const int64_t parsed = std::strtoll(text.c_str(), &end, 10);
  if (*end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool Validate(const Config& cfg) {
  if (cfg.ops.empty()) {
    std::printf("error: --ops needs at least one operation\n");
    return false;
  }
  for (const std::string& op : cfg.ops) {
    if (!IsValidOp(op)) {
      std::printf("error: unknown op '%s'\n", op.c_str());
      return false;
    }
  }
  if (cfg.gauss_size < 3 || cfg.gauss_size > 15 || cfg.gauss_size % 2 == 0) {
    std::printf("error: --gauss_size must be odd and in [3, 15]\n");
    return false;
  }
  if (cfg.canny_low < 0 || cfg.canny_high <= cfg.canny_low ||
      cfg.canny_high > 32767) {
    std::printf("error: need 0 <= --canny_low < --canny_high <= 32767\n");
    return false;
  }
  if (cfg.max_images < 0 || cfg.compare_height < 0 || cfg.device < 0) {
    std::printf(
        "error: --max_images, --compare_height and --device must be "
        ">= 0\n");
    return false;
  }
  return true;
}

}  // namespace

bool ParseArgs(int argc, char** argv, Config* cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      cfg->show_help = true;
      return true;
    }
    if (arg == "--no_compare") {
      cfg->save_compare = false;
      continue;
    }

    const size_t eq = arg.find('=');
    if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
      std::printf("error: bad argument '%s' (expected --name=value)\n",
                  arg.c_str());
      return false;
    }
    const std::string key = arg.substr(2, eq - 2);
    const std::string value = arg.substr(eq + 1);

    bool ok = true;
    if (key == "input_dir") {
      cfg->input_dir = value;
    } else if (key == "output_dir") {
      cfg->output_dir = value;
    } else if (key == "ops") {
      cfg->ops = SplitCommas(value);
    } else if (key == "gauss_size") {
      ok = ParseInt(value, &cfg->gauss_size);
    } else if (key == "canny_low") {
      ok = ParseInt(value, &cfg->canny_low);
    } else if (key == "canny_high") {
      ok = ParseInt(value, &cfg->canny_high);
    } else if (key == "max_images") {
      ok = ParseInt(value, &cfg->max_images);
    } else if (key == "compare_height") {
      ok = ParseInt(value, &cfg->compare_height);
    } else if (key == "device") {
      ok = ParseInt(value, &cfg->device);
    } else {
      std::printf("error: unknown flag --%s\n", key.c_str());
      return false;
    }
    if (!ok) {
      std::printf("error: bad value for --%s: '%s'\n", key.c_str(),
                  value.c_str());
      return false;
    }
  }
  return Validate(*cfg);
}

void PrintUsage(const char* program) {
  std::printf(
      "usage: %s [flags]\n"
      "\n"
      "  --input_dir=DIR       folder of png/jpg/bmp/pgm images "
      "(default data/sipi)\n"
      "  --output_dir=DIR      where results and logs go (default output)\n"
      "  --ops=a,b,c           ops applied in order on the gray image\n"
      "                        choose from equalize,gauss,sharpen,sobel,canny\n"
      "                        (default equalize,gauss,canny)\n"
      "  --gauss_size=N        gaussian mask size, odd 3..15 (default 5)\n"
      "  --canny_low=N         canny low threshold (default 60)\n"
      "  --canny_high=N        canny high threshold (default 150)\n"
      "  --max_images=N        stop after N images, 0 = all (default 0)\n"
      "  --compare_height=N    height of before|after images, 0 = full "
      "size (default 384)\n"
      "  --no_compare          skip the before|after images\n"
      "  --device=N            CUDA device id (default 0)\n"
      "  --help                show this message\n",
      program);
}

void PrintConfig(const Config& cfg) {
  std::string ops;
  for (const std::string& op : cfg.ops) {
    if (!ops.empty()) ops += " -> ";
    ops += op;
  }
  std::printf("input dir      : %s\n", cfg.input_dir.c_str());
  std::printf("output dir     : %s\n", cfg.output_dir.c_str());
  std::printf("pipeline       : gray -> %s\n", ops.c_str());
  std::printf("gauss size     : %d\n", cfg.gauss_size);
  std::printf("canny low/high : %d / %d\n", cfg.canny_low, cfg.canny_high);
  std::printf("max images     : %d\n", cfg.max_images);
  std::printf("compare images : %s (height %d)\n",
              cfg.save_compare ? "yes" : "no", cfg.compare_height);
  std::printf("\n");
}

}  // namespace npp_pipeline
