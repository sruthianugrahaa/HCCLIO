"""load_dataset.py - IoT tier: the camera replacement.

Reads the 1000-frame cobot dataset from ~/testbed/dataset/imagenet_1000/ and
yields one frame at a time as JPEG bytes with its ground-truth label.

    <root>/manifest.csv    image_file,ground_truth_idx,ground_truth_name
    <root>/images/*.jpg

Build the dataset first with download_dataset.py (same folder).

    python load_dataset.py            # prints how many frames and the first five
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "~/testbed/dataset/imagenet_1000"


@dataclass
class Frame:
    frame_id: int            # 0, 1, 2, ... in manifest order
    image_file: str          # path relative to the dataset root
    jpeg: bytes              # what the camera would have produced
    ground_truth_idx: int    # ImageNet-1k class index
    ground_truth_name: str   # e.g. "forklift"


def _to_jpeg(data: bytes, quality: int = 90) -> bytes:
    """Frames travel as JPEG over MQTT; re-encode anything that is not already JPEG."""
    if data[:2] == b"\xff\xd8":
        return data
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def load_dataset(root: str = DEFAULT_ROOT, limit: int | None = None, manifest: str = "manifest.csv") -> list[Frame]:
    """Return the dataset as a list of Frame objects (first `limit` frames if given)."""
    root = Path(os.path.expanduser(root))
    path = root / manifest
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run download_dataset.py first")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]
    frames = []
    for i, r in enumerate(rows):
        jpeg = _to_jpeg((root / r["image_file"]).read_bytes())
        frames.append(Frame(i, r["image_file"], jpeg, int(r["ground_truth_idx"]), r["ground_truth_name"]))
    return frames


if __name__ == "__main__":
    import sys

    frames = load_dataset(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    print(f"{len(frames)} frames")
    for f in frames[:5]:
        print(f.frame_id, f.image_file, f.ground_truth_idx, f.ground_truth_name, f"{len(f.jpeg)} bytes")
