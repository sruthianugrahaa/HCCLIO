"""backhaul_delay.py - Edge tier: simulated Edge <-> Cloud wired backhaul delay.

    T_bh ~ Gamma(m_bh, theta_bh)   [ms]
    m_bh     = floor((1 + 1.28 * M_BS/M_GW) * k1 + (h - 1) * k2)
    theta_bh = a + K_size * k3
    mean     = m_bh * theta_bh

Constants come from config/hcclio.yaml -> backhaul (placeholders, see README).

    python backhaul_delay.py     # prints m_bh, theta_bh and the delay statistics
"""

from __future__ import annotations

import math

import numpy as np


class BackhaulDelay:
    def __init__(self, params: dict, seed: int | None = None):
        p = {k: float(v) for k, v in params.items()}
        self.m_bh = max(1, math.floor((1.0 + 1.28 * p["mbs_over_mgw"]) * p["k1"] + (p["hops"] - 1.0) * p["k2"]))
        self.theta_ms = p["a_ms"] + p["k_size_bits"] * p["k3_ms_per_bit"]
        self.rng = np.random.default_rng(seed)

    def mean_ms(self) -> float:
        return self.m_bh * self.theta_ms

    def delay_ms(self) -> float:
        return float(self.rng.gamma(self.m_bh, self.theta_ms))


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "config" / "hcclio.yaml"))
    b = BackhaulDelay(cfg["backhaul"], seed=0)
    d = np.array([b.delay_ms() for _ in range(10000)])
    print(f"m_bh={b.m_bh}  theta={b.theta_ms:.3f} ms  mean={b.mean_ms():.2f} ms  "
          f"sampled mean={d.mean():.2f}  p95={np.percentile(d, 95):.2f} ms")
