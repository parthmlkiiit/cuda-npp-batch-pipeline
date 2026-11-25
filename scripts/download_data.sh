#!/usr/bin/env bash
# Downloads the USC-SIPI "misc", "textures", "aerials" and "sequences" volumes
# and converts every TIFF to PNG so the pipeline can read it.
set -euo pipefail

cd "$(dirname "$0")/.."

RAW_DIR=data/raw
OUT_DIR=data/sipi
VOLUMES="misc textures aerials sequences"
PYTHON=${PYTHON:-python3}

mkdir -p "$RAW_DIR" "$OUT_DIR"

for vol in $VOLUMES; do
  if [ ! -f "$RAW_DIR/$vol.tar.gz" ]; then
    echo "downloading $vol.tar.gz"
    curl -L --fail -o "$RAW_DIR/$vol.tar.gz" \
      "https://sipi.usc.edu/database/$vol.tar.gz"
  fi
  echo "extracting $vol.tar.gz"
  tar -xzf "$RAW_DIR/$vol.tar.gz" -C "$RAW_DIR"
done

if ! "$PYTHON" -c "import PIL" 2>/dev/null; then
  echo "Pillow not found, installing it for this user"
  "$PYTHON" -m pip install --user pillow
fi

"$PYTHON" scripts/convert_tiff.py "$RAW_DIR" "$OUT_DIR"
