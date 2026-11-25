"""Converts the SIPI TIFF files to 8-bit PNG (gray stays gray, rest -> RGB).

usage: python3 scripts/convert_tiff.py <raw_dir> <out_dir>
"""
import sys
from pathlib import Path

from PIL import Image


def to_8bit(img):
    if img.mode in ("L", "RGB"):
        return img
    if img.mode in ("I;16", "I;16B", "I"):
        return img.point(lambda v: v / 256).convert("L")
    if img.mode in ("1", "P", "LA"):
        return img.convert("L")
    return img.convert("RGB")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    raw_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    tiffs = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
    done = 0
    for path in tiffs:
        volume = path.relative_to(raw_dir).parts[0]
        target = out_dir / f"{volume}_{path.stem}.png"
        if target.exists():
            done += 1
            continue
        with Image.open(path) as img:
            to_8bit(img).save(target)
        done += 1
    print(f"converted {done} / {len(tiffs)} images into {out_dir}")


if __name__ == "__main__":
    main()
