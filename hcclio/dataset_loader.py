"""Dataset loader that replaces the picamera: yields JPEG frames from the 1000-image set.

Layout written by outputs/iot_device_tier/download_dataset.py:
    <root>/manifest.csv   image_file,ground_truth_idx,ground_truth_name
    <root>/images/*.jpg
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .classes import COBOT_CLASSES


@dataclass
class Frame:
    frame_id: int
    image_file: str
    jpeg: bytes
    ground_truth_idx: int
    ground_truth_name: str


def _ensure_jpeg(data: bytes, quality: int) -> bytes:
    if data[:2] == b"\xff\xd8":
        return data
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class DatasetLoader:
    def __init__(self, root: str | Path, manifest: str = "manifest.csv", limit: int | None = None,
                 jpeg_quality: int = 90):
        self.root = Path(root).expanduser()
        path = self.root / manifest
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run outputs/iot_device_tier/download_dataset.py first")
        with open(path, newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        if limit:
            self.rows = self.rows[:limit]
        self.quality = jpeg_quality

    def __len__(self):
        return len(self.rows)

    def __iter__(self) -> Iterator[Frame]:
        for i, r in enumerate(self.rows):
            data = (self.root / r["image_file"]).read_bytes()
            yield Frame(i, r["image_file"], _ensure_jpeg(data, self.quality),
                        int(r["ground_truth_idx"]), r["ground_truth_name"])


def write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_file", "ground_truth_idx", "ground_truth_name"])
        w.writeheader()
        w.writerows(rows)
    return path


def make_synthetic_dataset(root: str | Path, n: int = 100, seed: int = 0, size: int = 224) -> Path:
    """Noise images tagged with a ground truth in the JPEG comment, for stub-model dry runs."""
    from PIL import Image

    root = Path(root).expanduser()
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(COBOT_CLASSES)
    rows = []
    for i in range(n):
        name = names[i % len(names)]
        idx = COBOT_CLASSES[name]
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        fn = f"images/{i:04d}_{name}.jpg"
        Image.fromarray(arr).save(root / fn, format="JPEG", quality=85, comment=f"hcclio_gt={idx}".encode())
        rows.append({"image_file": fn, "ground_truth_idx": idx, "ground_truth_name": name})
    return write_manifest(root, rows)
