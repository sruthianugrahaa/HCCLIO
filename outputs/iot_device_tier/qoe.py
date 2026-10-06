"""qoe.py - per-frame QoE and the M/M/1 processing-rate (mu) model.

QoE (Ernest et al.)
    Q_x = (1 - T_x^E2E / T_i^comp) * A_x          x in {i, ES, CS}
    Q_x = 0 and tier = "Discard"                    if T_x^E2E >= T_i^comp
    A_x = pretrained accuracy of the tier that answered (0.75 / 0.80 / 0.85;
          Fallback-IoT uses A_i)

M/M/1 inference delay per tier (aligned with the cobot-tier model of the
earlier work; x = i (IoT), ES (Edge), CS (Cloud)):
    mu_x            = C_x * K_task / f_x            mean service time per frame [s]
                      C_x  computation complexity [ops/bit]
                      K_task input size [bit]
                      f_x  computational rate of tier x [ops/s]
    service rate    = 1 / mu_x                     [frames/s]
    lambda_x        = P_x * lambda                 arrivals that reach tier x
                      (P_i = 1, P_ES / P_CS = share of frames offloaded there)
    tau_sys,x       ~ Exp with mean 1 / (1/mu_x - P_x * lambda)   (stable if 1/mu_x > P_x lambda)
    tau_x           = tau_tx,x + tau_sys,x         transmission + system time
"""

from __future__ import annotations

import math

import numpy as np

# Pretrained accuracy per tier (config models.<tier>.accuracy overrides)
ACCURACY = {"IoT": 0.75, "Fallback-IoT": 0.75, "Edge": 0.80, "Cloud": 0.85}

# Forward-pass cost of the three ViTs at 224x224 (timm, GMACs per image)
VIT_GMACS = {"i": 4.6, "ES": 17.6, "CS": 61.6}


# ----------------------------------------------------------------- QoE
def qoe(e2e_ms: float, t_i_ms: float, a_x: float) -> tuple[float, bool]:
    """Return (Q_x, discarded)."""
    if e2e_ms >= t_i_ms:
        return 0.0, True
    return (1.0 - e2e_ms / t_i_ms) * a_x, False


def e2e_latency_ms(t: dict) -> float:
    """T_x^E2E = local + wireless + E2LM_edge + edge + backhaul + E2LM_cloud + cloud (missing = 0)."""
    keys = ("local_inference_ms", "wireless_delay_ms", "E2LM_edge_ms", "edge_inference_ms",
            "backhaul_delay_ms", "E2LM_cloud_ms", "cloud_inference_ms")
    return float(sum(float(t.get(k) or 0.0) for k in keys))


# ------------------------------------------------------- M/M/1 mu model
def service_time_s(c_ops_per_bit: float, k_task_bits: float, f_ops_per_s: float) -> float:
    """mu_x = C_x K_task / f_x."""
    return c_ops_per_bit * k_task_bits / f_ops_per_s


def complexity_from_model(tier: str, k_task_bits: float) -> float:
    """C_x [ops/bit] for the ViT of tier x, from its multiply-accumulate count."""
    return VIT_GMACS[tier] * 1e9 / k_task_bits


def rate_from_measurement(tier: str, mean_inference_ms: float) -> float:
    """f_x [ops/s] that reproduces the measured mean inference time (calibrates the model)."""
    return VIT_GMACS[tier] * 1e9 / (mean_inference_ms / 1000.0)


def mean_system_time_s(mu_s: float, p_x: float, lam: float) -> float:
    """tau_sys,x mean = (1/mu_x - P_x lambda)^-1, infinite when the queue is unstable."""
    slack = 1.0 / mu_s - p_x * lam
    return 1.0 / slack if slack > 0 else math.inf


def sample_system_time_ms(rng: np.random.Generator, mu_s: float, p_x: float, lam: float, size=None):
    m = mean_system_time_s(mu_s, p_x, lam)
    if math.isinf(m):
        return np.full(size, np.inf) if size is not None else math.inf
    return rng.exponential(m * 1000.0, size)


def tier_delay_ms(rng, tau_tx_ms: float, mu_s: float, p_x: float, lam: float) -> float:
    """tau_x = tau_tx + tau_sys,x."""
    return tau_tx_ms + float(sample_system_time_ms(rng, mu_s, p_x, lam))


if __name__ == "__main__":
    # Example: calibrate mu_i from a measured 180 ms ViT-Small time on the Pi 5
    K = 1e6
    f_i = rate_from_measurement("i", 180.0)
    mu_i = service_time_s(complexity_from_model("i", K), K, f_i)
    print(f"C_i={complexity_from_model('i', K):.0f} ops/bit  f_i={f_i:.3g} ops/s  mu_i={mu_i*1000:.1f} ms")
    for lam in (1, 3, 5):
        print(f"lambda={lam}/s  mean tau_sys,i = {mean_system_time_s(mu_i, 1.0, lam)*1000:.1f} ms")
