"""Simulated network delays: Rayleigh IoT<->Edge wireless link and Gamma wired backhaul."""

from __future__ import annotations

import math

import numpy as np


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def pathloss_los_db(fc_hz: float, d_m: float) -> float:
    return 31.84 + 19.0 * math.log10(fc_hz / 1e9) + 20.0 * math.log10(d_m)


def pathloss_nlos_db(fc_hz: float, d_m: float) -> float:
    return 32.4 + 20.0 * math.log10(fc_hz / 1e9) + 30.0 * math.log10(d_m)


def p_los(d_m: float, rho_c: float, d_c_m: float) -> float:
    """P_LOS(d) = exp(d * log(1 - rho_c) / d_c)."""
    return math.exp(d_m * math.log(1.0 - rho_c) / d_c_m)


class RayleighChannel:
    """IoT -> Edge uplink. |h|^2 ~ Exp(1), SNR = Pt|h|^2 / (N0 B PL), R = B log2(1+SNR)."""

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.fc = float(cfg["fc_hz"])
        self.B = float(cfg["bandwidth_hz"])
        self.d = float(cfg["distance_m"])
        self.pt_w = db_to_lin(float(cfg["pt_dbm"]) - 30.0)
        self.noise_w = db_to_lin(float(cfg["n0_dbm_per_hz"]) + float(cfg.get("noise_figure_db", 0.0)) - 30.0) * self.B
        self.payload = float(cfg["payload_bits"])
        self.p_los = p_los(self.d, float(cfg["rho_c"]), float(cfg["d_c_m"]))
        self.pl_los = db_to_lin(pathloss_los_db(self.fc, self.d))
        self.pl_nlos = db_to_lin(pathloss_nlos_db(self.fc, self.d))
        self.los_mode = cfg.get("los_mode", "sample")
        self.min_rate = float(cfg.get("min_rate_bps", 1.0))
        self.rng = rng or np.random.default_rng()

    def sample(self, payload_bits: float | None = None) -> dict:
        h2 = self.rng.exponential(1.0)
        if self.los_mode == "expected":
            pl = self.p_los * self.pl_los + (1.0 - self.p_los) * self.pl_nlos
            los = None
        else:
            los = bool(self.rng.random() < self.p_los)
            pl = self.pl_los if los else self.pl_nlos
        snr = self.pt_w * h2 / (self.noise_w * pl)
        rate = max(self.B * math.log2(1.0 + snr), self.min_rate)
        bits = self.payload if payload_bits is None else payload_bits
        return {"h2": h2, "los": los, "snr_db": 10 * math.log10(max(snr, 1e-30)),
                "rate_bps": rate, "tx_delay_ms": bits / rate * 1000.0}

    def delay_ms(self, payload_bits: float | None = None) -> float:
        return self.sample(payload_bits)["tx_delay_ms"]


class GammaBackhaul:
    """Edge -> Cloud wired backhaul. T_bh ~ Gamma(m_bh, theta_bh) [ms]."""

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def params(self, k_size_bits: float | None = None) -> tuple[int, float]:
        c = self.cfg
        k = float(c["k_size_bits"] if k_size_bits is None else k_size_bits)
        m_bh = math.floor((1.0 + 1.28 * float(c["mbs_over_mgw"])) * float(c["k1"])
                          + (int(c["hops"]) - 1) * float(c["k2"]))
        m_bh = max(m_bh, 1)
        theta = float(c["a_ms"]) + k * float(c["k3_ms_per_bit"])
        return m_bh, theta

    def mean_ms(self, k_size_bits: float | None = None) -> float:
        m, th = self.params(k_size_bits)
        return m * th

    def delay_ms(self, k_size_bits: float | None = None) -> float:
        m, th = self.params(k_size_bits)
        return float(self.rng.gamma(m, th))
