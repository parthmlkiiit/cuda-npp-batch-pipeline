#!/usr/bin/env bash
# One command for the lab: get the data (first time only), build, run on every
# image and pack output/ into proof_of_execution.tar.gz.
# Extra flags are passed to the program, e.g.
#   ./run.sh --ops=equalize,sobel --max_images=50
set -euo pipefail

cd "$(dirname "$0")"

if [ -z "$(ls -A data/sipi 2>/dev/null)" ]; then
  ./scripts/download_data.sh
fi

make

mkdir -p output
./bin/npp_batch --input_dir=data/sipi --output_dir=output "$@" \
  2>&1 | tee output/run.log

tar -czf proof_of_execution.tar.gz output
echo "packed output/ into proof_of_execution.tar.gz"
