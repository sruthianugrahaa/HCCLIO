"""edge_rayleigh_delay.py - Edge tier: simulated IoT <-> Edge wireless delay (Rayleigh fading).

The Edge draws one delay per frame and hands it to the IoT device in its E2LM
reply (see e2lm.py); the IoT device uses it in the latency gate.

Per frame:
    |h|^2 ~ Exp(1)                                  Rayleigh small-scale fading
    P_LOS(d) = exp(d * ln(1 - rho_c) / d_c)          LOS or NLOS drawn with this probability
    L_LOS  = 31.84 + 19 log10(fc/1e9) + 20 log10(d)  [dB]  (3GPP UMi-like)
    L_NLOS = 32.4  + 20 log10(fc/1e9) + 30 log10(d)  [dB]
    SNR = Pt |h|^2 / (N0 B PL)
    R   = B log2(1 + SNR)                            [bit/s]
    tx_delay_ms = payload / R * 1000                 payload = 1 Mb

    python edge_rayleigh_delay.py      # prints the delay statistics for 10 000 draws
"""

from __future__ import annotations

import math

import numpy as np

# spec values (overridden by config/hcclio.yaml -> wireless when main.py runs)
DEFAULTS = dict(fc_hz=3.5e9, bandwidth_hz=20e6, distance_m=10.0, pt_dbm=23.0, n0_dbm_per_hz=-174.0,
                noise_figure_db=0.0, payload_bits=1e6, rho_c=0.5, d_c_m=18.0)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def pathloss_db(fc_hz: float, d_m: float, los: bool) -> float:
    if los:
        return 31.84 + 19.0 * math.log10(fc_hz / 1e9) + 20.0 * math.log10(d_m)
    return 32.4 + 20.0 * math.log10(fc_hz / 1e9) + 30.0 * math.log10(d_m)


def p_los(d_m: float, rho_c: float, d_c_m: float) -> float:
    return math.exp(d_m * math.log(1.0 - rho_c) / d_c_m)


class EdgeRayleighDelay:
    def __init__(self, params: dict | None = None, seed: int | None = None):
        p = {k: float(v) for k, v in {**DEFAULTS, **(params or {})}.items() if k in DEFAULTS}
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.pt_w = db_to_linear(p["pt_dbm"] - 30.0)                                   # dBm -> W
        self.noise_w = db_to_linear(p["n0_dbm_per_hz"] + p["noise_figure_db"] - 30.0) * p["bandwidth_hz"]
        self.p_los = p_los(p["distance_m"], p["rho_c"], p["d_c_m"])

    def sample(self, payload_bits: float | None = None) -> dict:
        p = self.p
        h2 = self.rng.exponential(1.0)
        los = bool(self.rng.random() < self.p_los)
        pl = db_to_linear(pathloss_db(p["fc_hz"], p["distance_m"], los))
        snr = self.pt_w * h2 / (self.noise_w * pl)
        rate = max(p["bandwidth_hz"] * math.log2(1.0 + snr), 1.0)
        bits = p["payload_bits"] if payload_bits is None else payload_bits
        return {"h2": h2, "los": los, "snr_db": 10 * math.log10(max(snr, 1e-30)), "rate_bps": rate,
                "tx_delay_ms": bits / rate * 1000.0}

    def delay_ms(self) -> float:
        return self.sample()["tx_delay_ms"]


if __name__ == "__main__":
    ch = EdgeRayleighDelay(seed=0)
    d = np.array([ch.delay_ms() for _ in range(10000)])
    print(f"P_LOS={ch.p_los:.3f}  mean={d.mean():.2f} ms  median={np.median(d):.2f} ms  "
          f"p95={np.percentile(d, 95):.2f} ms  p99={np.percentile(d, 99):.2f} ms")
