"""Build the 1000-frame cobot dataset in ~/testbed/dataset/imagenet_1000/.

Images come from the ImageNet-1k validation split, restricted to the 30
cobot-relevant classes in hcclio/classes.py and balanced across them
(1000 / 30 -> 33 or 34 frames per class).

Sources (pick one):
  --source hf      Hugging Face `ILSVRC/imagenet-1k` (gated: accept the licence on
                   huggingface.co, then `huggingface-cli login`). Streams the split,
                   so no full download. Needs `pip install datasets`.
  --source dir     A local ImageNet folder laid out by synset, e.g.
                   /data/imagenet/val/n03085013/*.JPEG. Needs timm for index->synset.
  --source synthetic  Noise images for stub-model dry runs (no download).

    python outputs/iot_device_tier/download_dataset.py --source hf
    python outputs/iot_device_tier/download_dataset.py --source dir --dir /data/imagenet/val
"""
import argparse
import io
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from hcclio.classes import COBOT_CLASSES, INDEX_TO_NAME  # noqa: E402
from hcclio.config import load_config  # noqa: E402
from hcclio.dataset_loader import make_synthetic_dataset, write_manifest  # noqa: E402


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
    cfg = load_config()
    d = cfg["dataset"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "dir", "synthetic"], default="hf")
    ap.add_argument("--dir", type=pathlib.Path, help="ImageNet split root for --source dir")
    ap.add_argument("--out", default=d["root"])
    ap.add_argument("-n", type=int, default=int(d["n_frames"]))
    ap.add_argument("--seed", type=int, default=int(cfg["sim"]["seed"]))
    args = ap.parse_args(argv)

    root = pathlib.Path(args.out).expanduser()
    if args.source == "synthetic":
        print(f"wrote {make_synthetic_dataset(root, args.n, args.seed)}")
        return
    check_class_indices()
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
