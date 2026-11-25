// Copyright 2026 Parth
// Batch driver: runs the configured ops over every image in a folder,
// saves the results and logs per-image timings.

#ifndef SRC_PIPELINE_H_
#define SRC_PIPELINE_H_

#include <string>

#include "src/config.h"

namespace npp_pipeline {

// Timings and info for one processed image (one row of run_log.csv).
struct ImageStats {
  std::string file;
  int width = 0;
  int height = 0;
  int channels = 0;
  double load_ms = 0.0;      // disk -> host (decode)
  double upload_ms = 0.0;    // host -> device
  double gpu_ms = 0.0;       // all NPP work for this image
  double download_ms = 0.0;  // device -> host
  double save_ms = 0.0;      // host -> disk (encode)
  bool ok = false;
};

// Processes the whole input folder. Returns the number of images that failed,
// or -1 if nothing could be processed at all.
int RunBatch(const Config& cfg);

}  // namespace npp_pipeline

#endif  // SRC_PIPELINE_H_
