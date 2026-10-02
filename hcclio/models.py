"""Tier classifiers: real timm ViTs, or a seeded stub for dry runs and tests.

Every classifier maps JPEG bytes -> (softmax[1000] float32, inference_ms).
"""

from __future__ import annotations

import hashlib
import io
import time

import numpy as np

from .classes import COBOT_CLASSES

TIERS = ("iot", "edge", "cloud")


class Classifier:
    tier: str

    def predict(self, jpeg: bytes) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class TimmClassifier(Classifier):
    def __init__(self, tier: str, model_name: str, device: str = "auto", torch_threads: int = 0,
                 restrict_classes: list[int] | None = None):
        import timm
        import torch

        if torch_threads:
            torch.set_num_threads(torch_threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.tier = tier
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True).eval().to(device)
        dcfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**dcfg)
        self.mask = None
        if restrict_classes:
            mask = torch.full((self.model.num_classes,), float("-inf"))
            mask[restrict_classes] = 0.0
            self.mask = mask.to(device)
        self.predict(_blank_jpeg())  # warm-up so frame 0 is not an outlier

    def predict(self, jpeg):
        from PIL import Image

        torch = self.torch
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)[0]
            if self.mask is not None:
                logits = logits + self.mask
            probs = torch.softmax(logits.float(), dim=-1)
        probs = probs.cpu().numpy().astype(np.float32)
        return probs, (time.perf_counter() - t0) * 1000.0


def jpeg_ground_truth(jpeg: bytes) -> int | None:
    """Ground truth embedded by the synthetic dataset generator (JPEG comment)."""
    marker = b"hcclio_gt="
    i = jpeg.find(marker, 0, 4096)
    if i < 0:
        return None
    j = i + len(marker)
    k = j
    while k < len(jpeg) and jpeg[k:k + 1].isdigit():
        k += 1
    return int(jpeg[j:k]) if k > j else None


class StubClassifier(Classifier):
    """Deterministic simulated ViT: right with probability `accuracy`, confident when right.

    Needs no weights or torch. Reads ground truth from the synthetic dataset's JPEG
    comment; for other images it hashes the bytes onto a cobot class.
    """

    def __init__(self, tier: str, accuracy: float, latency_ms: float = 0.0, num_classes: int = 1000):
        self.tier = tier
        self.acc = accuracy
        self.latency_ms = latency_ms
        self.K = num_classes
        self._salt = TIERS.index(tier) if tier in TIERS else 7

    def predict(self, jpeg):
        t0 = time.perf_counter()
        digest = hashlib.sha256(jpeg).digest()
        seed = int.from_bytes(digest[:8], "little") ^ (self._salt * 0x9E3779B97F4A7C15)
        rng = np.random.default_rng(seed & (2**63 - 1))
        # shared per-image difficulty so tiers agree on hard frames
        difficulty = np.random.default_rng(int.from_bytes(digest[8:16], "little")).random()
        gt = jpeg_ground_truth(jpeg)
        if gt is None:
            gt = list(COBOT_CLASSES.values())[digest[0] % len(COBOT_CLASSES)]
        correct = rng.random() < self.acc * (1.15 - 0.3 * difficulty)
        if correct:
            pred, conf = gt, rng.beta(8, 2)
        else:
            pred = int(rng.integers(self.K - 1))
            pred = pred + 1 if pred >= gt else pred
            conf = rng.beta(2, 4)
        probs = np.full(self.K, 0.0, dtype=np.float64)
        rest = 1.0 - conf
        if not correct:
            probs[gt] = min(rest * 0.6, conf * 0.95)
            rest -= probs[gt]
        noise = rng.dirichlet(np.ones(self.K)) * rest
        noise[pred] = 0.0
        if not correct:
            noise[gt] = 0.0
        noise *= rest / max(noise.sum(), 1e-12)
        probs += noise
        probs[pred] = conf
        probs /= probs.sum()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        return probs.astype(np.float32), (time.perf_counter() - t0) * 1000.0


def _blank_jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224)).save(buf, format="JPEG")
    return buf.getvalue()


def build_classifier(cfg, tier: str) -> Classifier:
    m = cfg["models"]
    if m["backend"] == "stub":
        return StubClassifier(tier, float(m[tier]["accuracy"]),
                              float(m["stub"]["latency_ms"].get(tier, 0)), int(m["stub"]["num_classes"]))
    restrict = list(COBOT_CLASSES.values()) if m.get("restrict_to_dataset_classes") else None
    return TimmClassifier(tier, m[tier]["name"], m.get("device", "auto"), int(m.get("torch_threads", 0)), restrict)
