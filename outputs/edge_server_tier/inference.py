"""inference.py - Edge tier: ViT-Base inference, weighted ensemble and confidence gate.

For a frame offloaded by the IoT device (JPEG + p_i):
    p_ES, t_ES  = ViT-Base(JPEG)
    p_ens       = w_i * p_i + w_ES * p_ES          (0.44 / 0.56)
    c_edge      = max(p_ens),  y_edge = argmax(p_ens)
    c_edge >= tau_conf (0.80)  ->  answer at the Edge
    otherwise                  ->  cascade to the Cloud with p_i and p_ES

edge_only benchmark: no p_i, ViT-Base decides alone and always answers.
DCI benchmark with dci.use_ensemble=false: ViT-Base decides alone, may cascade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeResult:
    p_ES: np.ndarray
    edge_inference_ms: float
    c_edge: float          # aggregated confidence used by the gate
    y_edge: int            # prediction at the Edge
    answer_here: bool      # True -> reply to the IoT device, False -> cascade to the Cloud


def weighted_ensemble(probs: list[np.ndarray], weights: list[float]) -> tuple[np.ndarray, float, int]:
    p = sum(w * np.asarray(q, dtype=np.float64) for q, w in zip(probs, weights))
    y = int(np.argmax(p))
    return p, float(p[y]), y


def edge_inference(model, jpeg: bytes, p_i: np.ndarray | None, strategy: str, cfg) -> EdgeResult:
    p_es, t_ms = model.predict(jpeg)
    if p_i is None or strategy == "edge_only":
        y = int(np.argmax(p_es))
        return EdgeResult(p_es, t_ms, float(p_es[y]), y, answer_here=True)
    if strategy == "dci" and not cfg.dci_use_ensemble:
        y = int(np.argmax(p_es))
        c = float(p_es[y])
    else:
        _, c, y = weighted_ensemble([p_i, p_es], [cfg.w_i, cfg.w_ES])
    return EdgeResult(p_es, t_ms, c, y, answer_here=c >= cfg.tau_conf)
