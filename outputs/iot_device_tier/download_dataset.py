"""Build the 1000-frame cobot dataset in ~/testbed/dataset/imagenet_1000/.

Images come from the ImageNet-1k validation split, restricted to the 30
cobot-relevant classes in common/classes.py and balanced across them
(1000 / 30 -> 33 or 34 frames per class).

Sources (pick one):
  --source hf      Hugging Face `ILSVRC/imagenet-1k` (gated: accept the licence on
                   huggingface.co, then `huggingface-cli login`). Streams the split,
                   so no full download. Needs `pip install datasets`.
  --source dir     A local ImageNet folder laid out by synset, e.g.
                   /data/imagenet/val/n03085013/*.JPEG. Needs timm for index->synset.
  --source existing  Images you already have in one folder (any layout). Writes
                   manifest.csv next to them without copying anything. The class of
                   each image is read from its sub-folder or file name (synset such
                   as n03384352, or a class name such as forklift), or from
                   --labels CSV (columns: file name, then label as synset, class
                   index or class name; Kaggle's LOC_val_solution.csv also works).
  --source synthetic  Noise images for stub-model dry runs (no download).

    python outputs/iot_device_tier/download_dataset.py --source hf
    python outputs/iot_device_tier/download_dataset.py --source dir --dir /data/imagenet/val
    python outputs/iot_device_tier/download_dataset.py --source existing --dir ~/HCCLIO/images
"""
import argparse
import io
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # outputs/ (holds common/)

from common.classes import COBOT_CLASSES, INDEX_TO_NAME  # noqa: E402
from common.settings import load_settings  # noqa: E402


def write_manifest(root: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    import csv

    path = root / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_file", "ground_truth_idx", "ground_truth_name"])
        w.writeheader()
        w.writerows(rows)
    return path


def make_synthetic_dataset(root, n: int = 100, seed: int = 0, size: int = 224) -> pathlib.Path:
    """Noise images tagged with their ground truth in the JPEG comment (read by the stub ViTs)."""
    import numpy as np
    from PIL import Image

    root = pathlib.Path(root).expanduser()
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(COBOT_CLASSES)
    rows = []
    for i in range(n):
        name = names[i % len(names)]
        idx = COBOT_CLASSES[name]
        fn = f"images/{i:04d}_{name}.jpg"
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(
            root / fn, format="JPEG", quality=85, comment=f"hcclio_gt={idx}".encode())
        rows.append({"image_file": fn, "ground_truth_idx": idx, "ground_truth_name": name})
    return write_manifest(root, rows)


def quotas(n: int) -> dict[int, int]:
    idxs = sorted(COBOT_CLASSES.values())
    base, extra = divmod(n, len(idxs))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(idxs)}


def save_jpeg(img, path: pathlib.Path, quality: int):
    img.convert("RGB").save(path, format="JPEG", quality=quality)


def from_hf(root, n, quality, dataset_id, split):
    from datasets import load_dataset

    want = quotas(n)
    have = {c: 0 for c in want}
    rows = []
    ds = load_dataset(dataset_id, split=split, streaming=True)
    for ex in ds:
        c = int(ex["label"])
        if c not in want or have[c] >= want[c]:
            continue
        name = INDEX_TO_NAME[c]
        fn = f"images/{name}_{have[c]:03d}.jpg"
        save_jpeg(ex["image"], root / fn, quality)
        rows.append({"image_file": fn, "ground_truth_idx": c, "ground_truth_name": name})
        have[c] += 1
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{n}")
        if len(rows) >= n:
            break
    return rows


def from_dir(root, n, quality, src: pathlib.Path, seed: int):
    from PIL import Image
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    want = quotas(n)
    rng = random.Random(seed)
    rows = []
    for c, k in want.items():
        name = INDEX_TO_NAME[c]
        files = sorted(p for p in (src / info.index_to_label_name(c)).glob("*") if p.is_file())
        if len(files) < k:
            print(f"warning: {name} has only {len(files)} images (wanted {k})")
        for j, p in enumerate(rng.sample(files, min(k, len(files)))):
            fn = f"images/{name}_{j:03d}.jpg"
            save_jpeg(Image.open(p), root / fn, quality)
            rows.append({"image_file": fn, "ground_truth_idx": c, "ground_truth_name": name})
    return rows


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _label_lookup():
    """Map synset / class name / index text -> (ImageNet-1k index, name)."""
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    by_key = {}
    for i in range(1000):
        name = INDEX_TO_NAME.get(i) or info.index_to_description(i).split(",")[0].strip().replace(" ", "_")
        by_key[info.index_to_label_name(i).lower()] = (i, name)
        by_key[str(i)] = (i, name)
    for name, i in COBOT_CLASSES.items():
        by_key[name.lower()] = (i, name)
    return by_key


def _read_labels(path: pathlib.Path) -> dict[str, str]:
    """file stem -> label text, from a 2-column CSV or Kaggle's LOC_val_solution.csv."""
    import csv

    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0].lower() in ("imageid", "file", "filename", "image_file"):
                continue
            out[pathlib.Path(row[0]).stem.lower()] = row[1].split()[0]  # Kaggle: "n01751748 x y x y ..."
    return out


def _guess_label(rel: pathlib.Path, by_key: dict, labels: dict):
    import re

    if rel.stem.lower() in labels:
        return by_key.get(labels[rel.stem.lower()].lower())
    text = "/".join(rel.parts).lower()
    m = re.search(r"n\d{8}", text)
    if m and m.group(0) in by_key:
        return by_key[m.group(0)]
    norm = re.sub(r"[\s\-]+", "_", text)
    for name in sorted(COBOT_CLASSES, key=len, reverse=True):  # longest first: pop_bottle before bottle
        if re.search(rf"(^|[/_.]){re.escape(name)}([/_.\d]|$)", norm):
            return by_key[name]
    return None


def from_existing(src: pathlib.Path, labels_csv: pathlib.Path | None):
    """Label images already on disk; returns (root, rows) with paths relative to root."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"folder not found: {src}")
    root = src.parent
    by_key = _label_lookup()
    labels = _read_labels(labels_csv.expanduser()) if labels_csv else {}
    rows, unknown = [], []
    for p in sorted(q for q in src.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_EXTS):
        hit = _guess_label(p.relative_to(src), by_key, labels)
        if hit is None:
            unknown.append(p)
            continue
        rows.append({"image_file": p.relative_to(root).as_posix(), "ground_truth_idx": hit[0],
                     "ground_truth_name": hit[1]})
    if not rows and not unknown:
        found = sorted(p.name for p in src.iterdir())[:5]
        sys.exit(f"no image files ({', '.join(sorted(IMAGE_EXTS))}) in {src}"
                 + (f"; it contains e.g. {found}" if found else "; the folder is empty"))
    other = sum(1 for r in rows if r["ground_truth_idx"] not in INDEX_TO_NAME)
    print(f"{len(rows)} images labelled, {len(unknown)} without a label (skipped)"
          + (f", {other} labelled with a class outside the 30 cobot classes" if other else ""))
    for p in unknown[:5]:
        print(f"  no label for {p.relative_to(src)}")
    if unknown and not labels:
        print("  -> pass --labels <csv> mapping each file name to its synset or class")
    return root, rows


def check_class_indices():
    """Cross-check our index table against timm's ImageNet-1k descriptions, when timm is present."""
    try:
        from timm.data import ImageNetInfo
    except ImportError:
        return
    info = ImageNetInfo()
    for name, idx in COBOT_CLASSES.items():
        desc = info.index_to_description(idx).lower()
        key = name.replace("_", " ").split()[0]
        if key not in desc and name not in ("plane", "screen", "mouse", "pop_bottle", "cellular_telephone"):
            print(f"warning: index {idx} is '{desc}', expected {name}")


def main(argv=None):
    cfg = load_settings()
    d = cfg["dataset"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "dir", "existing", "synthetic"], default="hf")
    ap.add_argument("--dir", type=pathlib.Path, help="ImageNet split root (--source dir) or your images folder (--source existing)")
    ap.add_argument("--labels", type=pathlib.Path, help="--source existing: CSV of file name -> synset / index / class name")
    ap.add_argument("--out", default=d["root"])
    ap.add_argument("-n", type=int, default=int(d["n_frames"]))
    ap.add_argument("--seed", type=int, default=int(cfg["sim"]["seed"]))
    args = ap.parse_args(argv)

    root = pathlib.Path(args.out).expanduser()
    if args.source == "synthetic":
        print(f"wrote {make_synthetic_dataset(root, args.n, args.seed)}")
        return
    check_class_indices()
    if args.source == "existing":
        if not args.dir:
            sys.exit("--source existing needs --dir <your images folder>")
        root, rows = from_existing(args.dir, args.labels)
        if not rows:
            sys.exit("no labelled images found")
        random.Random(args.seed).shuffle(rows)
        print(f"{len(rows)} frames -> {write_manifest(root, rows)}")
        print(f"run the IoT tier with:  --dataset {root}")
        return
    (root / "images").mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        rows = from_hf(root, args.n, int(d["jpeg_quality"]), d["hf_dataset"], d["hf_split"])
    else:
        if not args.dir:
            sys.exit("--source dir needs --dir")
        rows = from_dir(root, args.n, int(d["jpeg_quality"]), args.dir, args.seed)
    random.Random(args.seed).shuffle(rows)  # interleave classes in frame order
    print(f"{len(rows)} frames -> {write_manifest(root, rows)}")


if __name__ == "__main__":
    main()
