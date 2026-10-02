"""Weighted collaborative ensembles, QoE (Ernest et al.) and the per-frame CSV schema."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

import numpy as np

CSV_COLUMNS = [
    "frame_id", "timestamp", "image_file", "ground_truth_idx", "ground_truth_name",
    "tier", "route", "path", "prediction_idx", "class_name", "correct",
    "iot_confidence", "aggregated_conf",
    "local_inference_ms", "wireless_delay_ms", "E2LM_edge_ms", "edge_inference_ms",
    "backhaul_delay_ms", "E2LM_cloud_ms", "cloud_inference_ms",
    "e2e_latency_ms", "A_x", "Q_x", "tau_conf", "tau_lat_ms", "T_i_ms",
]

# Delay components summed into e2e_latency_ms (missing ones count as 0).
DELAY_COLUMNS = [
    "local_inference_ms", "wireless_delay_ms", "E2LM_edge_ms", "edge_inference_ms",
    "backhaul_delay_ms", "E2LM_cloud_ms", "cloud_inference_ms",
]

TIER_ACCURACY_KEY = {"IoT": "iot", "Fallback-IoT": "iot", "Edge": "edge", "Cloud": "cloud"}


def ensemble(probs: list[np.ndarray], weights: list[float]) -> tuple[np.ndarray, float, int]:
    """p_ens = sum_k w_k p_k ; returns (p_ens, max(p_ens), argmax(p_ens))."""
    p = np.zeros_like(probs[0], dtype=np.float64)
    for pk, wk in zip(probs, weights):
        p += wk * np.asarray(pk, dtype=np.float64)
    y = int(np.argmax(p))
    return p, float(p[y]), y


def top1(p: np.ndarray) -> tuple[float, int]:
    y = int(np.argmax(p))
    return float(p[y]), y


def e2e_latency(timings: dict) -> float:
    return float(sum(float(timings.get(k) or 0.0) for k in DELAY_COLUMNS))


def qoe(e2e_ms: float, t_i_ms: float, a_x: float) -> tuple[float, bool]:
    """Q_x = (1 - T_x^E2E / T_i^comp) * A_x, or (0, discarded) if T_x^E2E >= T_i^comp."""
    if e2e_ms >= t_i_ms:
        return 0.0, True
    return (1.0 - e2e_ms / t_i_ms) * a_x, False


def qoe_vec(e2e_ms: np.ndarray, t_i_ms, a_x) -> np.ndarray:
    e2e_ms = np.asarray(e2e_ms, dtype=float)
    q = (1.0 - e2e_ms / t_i_ms) * a_x
    return np.where(e2e_ms >= t_i_ms, 0.0, q)


def tier_accuracy(cfg, tier: str) -> float:
    return float(cfg["models"][TIER_ACCURACY_KEY[tier]]["accuracy"])


class CsvLogger:
    def __init__(self, path: str | os.PathLike, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not (append and self.path.exists())
        self._fh = open(self.path, "w" if new else "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            self._w.writeheader()
        self._lock = threading.Lock()

    def log(self, row: dict) -> None:
        out = {}
        for k in CSV_COLUMNS:
            v = row.get(k, "")
            out[k] = f"{v:.4f}" if isinstance(v, float) else v
        with self._lock:
            self._w.writerow(out)
            self._fh.flush()

    def close(self):
        self._fh.close()
