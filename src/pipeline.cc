// Copyright 2026 Parth
#include "src/pipeline.h"

#include <cuda_runtime_api.h>
#include <npp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "src/config.h"
#include "src/gpu_image.h"
#include "src/image_io.h"
#include "src/npp_ops.h"

namespace npp_pipeline {
namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Times work queued on one CUDA stream with a pair of events.
class GpuTimer {
 public:
  explicit GpuTimer(cudaStream_t stream) : stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void Start() { cudaEventRecord(start_, stream_); }

  double StopMs() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct Totals {
  int ok = 0;
  int failed = 0;
  double megapixels = 0.0;
  double load_ms = 0.0;
  double upload_ms = 0.0;
  double gpu_ms = 0.0;
  double download_ms = 0.0;
  double save_ms = 0.0;
  // The first image pays for CUDA/NPP lazy loading, so keep it apart.
  double first_gpu_ms = 0.0;
  double first_megapixels = 0.0;
};

bool ProcessImage(const Config& cfg, const std::string& path,
                  const NppStreamContext& ctx, GpuTimer* timer,
                  ImageStats* stats) {
  const std::string stem = FileStem(path);
  stats->file = stem;

  Clock::time_point start = Clock::now();
  HostImage input;
  if (!LoadImage(path, &input)) return false;
  stats->load_ms = MsSince(start);
  stats->width = input.width;
  stats->height = input.height;
  stats->channels = input.channels;

  const int width = input.width;
  const int height = input.height;

  // Size of the before|after picture, keeps the aspect ratio.
  int thumb_height = height;
  int thumb_width = width;
  if (cfg.save_compare && cfg.compare_height > 0 &&
      cfg.compare_height < height) {
    thumb_height = cfg.compare_height;
    thumb_width = std::max(
        1, static_cast<int>(std::lround(1.0 * width * thumb_height / height)));
  }
  const bool shrink = thumb_height != height;

  // All device memory for this image is allocated before the timers run.
  GpuImage d_input(width, height, input.channels);
  GpuImage ping(width, height, 1);
  GpuImage pong(width, height, 1);
  std::unique_ptr<GpuImage> d_gray;
  std::unique_ptr<GpuImage> before_small;
  std::unique_ptr<GpuImage> after_small;
  if (input.channels == 3) {
    d_gray = std::make_unique<GpuImage>(width, height, 1);
  }
  if (shrink) {
    before_small = std::make_unique<GpuImage>(thumb_width, thumb_height, 1);
    after_small = std::make_unique<GpuImage>(thumb_width, thumb_height, 1);
  }
  if (!d_input.ok() || !ping.ok() || !pong.ok() || (d_gray && !d_gray->ok()) ||
      (before_small && (!before_small->ok() || !after_small->ok()))) {
    return false;
  }

  timer->Start();
  if (!d_input.Upload(input)) return false;
  stats->upload_ms = timer->StopMs();

  timer->Start();
  const GpuImage* gray = &d_input;
  if (d_gray) {
    if (!RgbToGray(d_input, d_gray.get(), ctx)) return false;
    gray = d_gray.get();
  }

  // Ping-pong between two buffers so the gray "before" image stays intact.
  const GpuImage* src = gray;
  GpuImage* dst = &ping;
  for (const std::string& op : cfg.ops) {
    if (!RunOp(op, cfg, *src, dst, ctx)) {
      std::printf("  op '%s' failed on %s\n", op.c_str(), stem.c_str());
      return false;
    }
    src = dst;
    dst = (dst == &ping) ? &pong : &ping;
  }
  const GpuImage& result = *src;

  const GpuImage* before_view = gray;
  const GpuImage* after_view = &result;
  if (shrink) {
    if (!Resize(*gray, before_small.get(), ctx) ||
        !Resize(result, after_small.get(), ctx)) {
      return false;
    }
    before_view = before_small.get();
    after_view = after_small.get();
  }
  if (!CudaOk(cudaGetLastError(), "pipeline")) return false;
  stats->gpu_ms = timer->StopMs();

  timer->Start();
  HostImage after;
  HostImage before_host;
  HostImage after_host;
  bool ok = result.Download(&after);
  if (ok && cfg.save_compare) {
    ok = before_view->Download(&before_host) &&
         after_view->Download(&after_host);
  }
  stats->download_ms = timer->StopMs();
  if (!ok) return false;

  start = Clock::now();
  ok = SavePng(cfg.output_dir + "/images/" + stem + "_after.png", after);
  if (ok && cfg.save_compare) {
    ok = SavePng(cfg.output_dir + "/compare/" + stem + "_before_after.png",
                 SideBySide(before_host, after_host));
  }
  stats->save_ms = MsSince(start);
  return ok;
}

void PrintSummary(const Totals& totals, double wall_ms,
                  const std::string& csv_path) {
  const double gpu_s = totals.gpu_ms / 1000.0;
  const double mp_per_s = gpu_s > 0.0 ? totals.megapixels / gpu_s : 0.0;
  const double warm_ms = totals.gpu_ms - totals.first_gpu_ms;
  const double warm_mp = totals.megapixels - totals.first_megapixels;
  const double warm_mp_per_s =
      warm_ms > 0.0 ? warm_mp / (warm_ms / 1000.0) : 0.0;
  std::printf("\n================ summary ================\n");
  std::printf("images ok / failed : %d / %d\n", totals.ok, totals.failed);
  std::printf("pixels processed   : %.1f MP\n", totals.megapixels);
  std::printf("disk read (decode) : %10.1f ms\n", totals.load_ms);
  std::printf("host -> device     : %10.1f ms\n", totals.upload_ms);
  std::printf("GPU (NPP) compute  : %10.1f ms  (%.0f MP/s)\n", totals.gpu_ms,
              mp_per_s);
  std::printf("  first image only : %10.1f ms  (CUDA/NPP warm-up)\n",
              totals.first_gpu_ms);
  std::printf("  all the rest     : %10.1f ms  (%.0f MP/s)\n", warm_ms,
              warm_mp_per_s);
  std::printf("device -> host     : %10.1f ms\n", totals.download_ms);
  std::printf("disk write (encode): %10.1f ms\n", totals.save_ms);
  std::printf("wall time          : %10.1f ms\n", wall_ms);
  std::printf("per-image log      : %s\n", csv_path.c_str());
}

}  // namespace

int RunBatch(const Config& cfg) {
  std::vector<std::string> files = ListImages(cfg.input_dir);
  if (files.empty()) {
    std::printf("error: no images found in %s\n", cfg.input_dir.c_str());
    return -1;
  }
  if (cfg.max_images > 0 &&
      files.size() > static_cast<size_t>(cfg.max_images)) {
    files.resize(cfg.max_images);
  }
  if (!MakeDirs(cfg.output_dir + "/images") ||
      (cfg.save_compare && !MakeDirs(cfg.output_dir + "/compare"))) {
    return -1;
  }

  const std::string csv_path = cfg.output_dir + "/run_log.csv";
  std::FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (csv == nullptr) {
    std::printf("error: cannot open %s\n", csv_path.c_str());
    return -1;
  }
  std::fprintf(csv,
               "index,file,width,height,channels,load_ms,upload_ms,gpu_ms,"
               "download_ms,save_ms,status\n");

  NppStreamContext ctx;
  if (!NppOk(nppGetStreamContext(&ctx), "nppGetStreamContext")) {
    std::fclose(csv);
    return -1;
  }
  GpuTimer timer(ctx.hStream);

  std::printf("processing %zu images from %s\n\n", files.size(),
              cfg.input_dir.c_str());
  Totals totals;
  const Clock::time_point batch_start = Clock::now();
  const int count = static_cast<int>(files.size());

  for (int i = 0; i < count; ++i) {
    ImageStats stats;
    stats.ok = ProcessImage(cfg, files[i], ctx, &timer, &stats);

    std::printf(
        "[%3d/%3d] %-28s %4dx%-4d x%d  up %6.2f  gpu %7.2f  down %6.2f ms"
        "  %s\n",
        i + 1, count, stats.file.c_str(), stats.width, stats.height,
        stats.channels, stats.upload_ms, stats.gpu_ms, stats.download_ms,
        stats.ok ? "ok" : "FAILED");
    std::fprintf(csv, "%d,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n", i + 1,
                 stats.file.c_str(), stats.width, stats.height, stats.channels,
                 stats.load_ms, stats.upload_ms, stats.gpu_ms,
                 stats.download_ms, stats.save_ms, stats.ok ? "ok" : "failed");

    if (!stats.ok) {
      ++totals.failed;
      continue;
    }
    const double megapixels =
        stats.width * static_cast<double>(stats.height) / 1e6;
    if (totals.ok == 0) {
      totals.first_gpu_ms = stats.gpu_ms;
      totals.first_megapixels = megapixels;
    }
    ++totals.ok;
    totals.megapixels += megapixels;
    totals.load_ms += stats.load_ms;
    totals.upload_ms += stats.upload_ms;
    totals.gpu_ms += stats.gpu_ms;
    totals.download_ms += stats.download_ms;
    totals.save_ms += stats.save_ms;
  }
  std::fclose(csv);

  PrintSummary(totals, MsSince(batch_start), csv_path);
  return totals.failed;
}

}  // namespace npp_pipeline
