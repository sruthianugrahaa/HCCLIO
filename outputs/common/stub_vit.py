"""stub_vit.py - simulated ViT used for dry runs and tests (backend: stub). numpy only.

Right with probability `accuracy`, confident when right, deterministic per image.
Reads the ground truth that download_dataset.py --source synthetic writes into the
JPEG comment; for real images it hashes the bytes onto a cobot class.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .classes import COBOT_CLASSES

TIERS = ("iot", "edge", "cloud")


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


class StubClassifier:
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
