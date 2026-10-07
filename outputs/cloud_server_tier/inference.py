"""inference.py - Cloud tier: ViT-Large inference, weighted ensemble and final decision.

    p_CS, t_CS = ViT-Large(JPEG)
    cascade from the Edge (p_i, p_ES):   p_ens = 0.28 p_i + 0.34 p_ES + 0.38 p_CS
    direct from the IoT   (p_i only):    p_ens = 0.424 p_i + 0.576 p_CS   (0.28 / 0.38 renormalised)
    c_cloud = max(p_ens), y_cloud = argmax(p_ens)
    c_cloud >= tau_conf (0.80)  ->  tier "Cloud",        result y_cloud
    otherwise                   ->  tier "Fallback-IoT", result y_i = argmax(p_i)

cloud_only benchmark: no p_i, ViT-Large decides alone and always answers (tier "Cloud").
DCI benchmark with dci.use_ensemble=false: ViT-Large decides alone, same gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CloudResult:
    p_CS: np.ndarray
    cloud_inference_ms: float
    tier: str              # "Cloud" or "Fallback-IoT"
    prediction: int
    c_cloud: float         # aggregated confidence used by the gate
    gate: str              # route suffix: "" | "|cloud_conf_pass" | "|cloud_conf_fail"


def weighted_ensemble(probs: list[np.ndarray], weights: list[float]) -> tuple[float, int]:
    p = sum(w * np.asarray(q, dtype=np.float64) for q, w in zip(probs, weights))
    y = int(np.argmax(p))
    return float(p[y]), y


def cloud_inference(model, msg: dict, cfg) -> CloudResult:
    p_cs, t_ms = model.predict(msg["jpeg"])
    p_i, p_es = msg.get("p_i"), msg.get("p_es")

    if p_i is None or msg["strategy"] == "cloud_only":
        y = int(np.argmax(p_cs))
        return CloudResult(p_cs, t_ms, "Cloud", y, float(p_cs[y]), "")

    if msg["strategy"] == "dci" and not cfg.dci_use_ensemble:
        y = int(np.argmax(p_cs))
        c = float(p_cs[y])
    elif p_es is not None:  # cascade from the Edge
        w = cfg.w_cascade
        c, y = weighted_ensemble([p_i, p_es, p_cs], [w["w_i"], w["w_ES"], w["w_CS"]])
    else:  # direct from the IoT device
        w = cfg.w_direct
        c, y = weighted_ensemble([p_i, p_cs], [w["w_i"], w["w_CS"]])

    if c >= cfg.tau_conf:
        return CloudResult(p_cs, t_ms, "Cloud", y, c, "|cloud_conf_pass")
    return CloudResult(p_cs, t_ms, "Fallback-IoT", int(np.argmax(p_i)), c, "|cloud_conf_fail")
