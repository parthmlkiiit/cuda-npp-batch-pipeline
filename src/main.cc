// Copyright 2026 Parth
// npp_batch: GPU batch image enhancement + edge detection with CUDA NPP.
//
//   ./bin/npp_batch --input_dir=data/sipi --ops=equalize,gauss,canny
//
// See README.md for every flag.

#include <cstdio>

#include "src/config.h"
#include "src/gpu_image.h"
#include "src/pipeline.h"

int main(int argc, char** argv) {
  npp_pipeline::Config cfg;
  if (!npp_pipeline::ParseArgs(argc, argv, &cfg)) {
    std::printf("\n");
    npp_pipeline::PrintUsage(argv[0]);
    return 1;
  }
  if (cfg.show_help) {
    npp_pipeline::PrintUsage(argv[0]);
    return 0;
  }

  std::printf("==== npp_batch: CUDA NPP batch image pipeline ====\n");
  if (!npp_pipeline::SelectDevice(cfg.device)) return 1;
  npp_pipeline::PrintConfig(cfg);

  const int failed = npp_pipeline::RunBatch(cfg);
  if (failed < 0) return 1;
  return failed == 0 ? 0 : 2;
}
