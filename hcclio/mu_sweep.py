"""Monte-Carlo sweep of QoE vs the Edge (or Cloud) processing rate mu_x on logged CSVs.

For each mu and each trial, the tier's logged inference time is replaced by a
sampled service/sojourn time, e2e latency is recomputed from the other logged
components, and Q_x is re-evaluated (Discard when e2e >= T_i^comp). Routing and
predictions are kept as logged, because they depend on confidences, not on mu.

mu here is the service RATE of tier x in frames/s, i.e. 1 / mu_x where mu_x =
C_x K_task / f_x is the mean service time of the M/M/1 cobot-tier model.

  service_model mm1 : system time ~ Exp(mean 1 / (mu - P_x * lambda))
                      P_x = share of frames that reach tier x in the log
                      (P_i = 1); mu <= P_x * lambda -> unstable queue, Discard
  service_model exp : service ~ Exp(mu)
  service_model det : service = 1 / mu
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .qoe import DELAY_COLUMNS

TIER_COLUMN = {"i": "local_inference_ms", "ES": "edge_inference_ms", "CS": "cloud_inference_ms"}
STRATEGY_LABELS = {
    "qoe_coclio.csv": "HCCLIO",
    "qoe_distributed_benchmark.csv": "DCI (conf. only)",
    "qoe_edge_only.csv": "Edge only",
    "qoe_cloud_only.csv": "Cloud only",
    "qoe_local_only.csv": "Local only",
}


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {c: np.array([_f(r.get(c)) for r in rows]) for c in DELAY_COLUMNS + ["A_x", "T_i_ms", "correct"]}
    out["path"] = np.array([r.get("path", "") for r in rows])
    return out


def sample_service_ms(rng, mu: float, n_trials: int, n_rows: int, model: str, lam: float) -> np.ndarray:
    shape = (n_trials, n_rows)
    if model == "det":
        return np.full(shape, 1000.0 / mu)
    if model == "exp":
        return rng.exponential(1000.0 / mu, shape)
    if model == "mm1":
        if mu <= lam:
            return np.full(shape, np.inf)
        return rng.exponential(1000.0 / (mu - lam), shape)
    raise ValueError(f"unknown service_model {model}")


def sweep(rows: dict, mu_values, tier: str = "ES", trials: int = 500, model: str = "mm1",
          arrival_rate: float = 1.0, seed: int = 0, t_i_ms: float | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    col = TIER_COLUMN[tier]
    if tier == "i":
        uses = rows[col] > 0  # every frame that ran ViT-Small locally
    else:
        uses = np.char.find(rows["path"].astype(str), "Edge" if tier == "ES" else "Cloud") >= 0
    p_x = float(uses.mean())  # P_x: share of frames that reach tier x
    base = sum(rows[c] for c in DELAY_COLUMNS) - np.where(uses, rows[col], 0.0)
    t_i = np.full(base.shape, t_i_ms) if t_i_ms else rows["T_i_ms"]
    a_x = rows["A_x"]
    out = []
    for mu in mu_values:
        svc = sample_service_ms(rng, float(mu), trials, int(uses.sum()), model, p_x * arrival_rate)
        e2e = np.tile(base, (trials, 1))
        e2e[:, uses] += svc
        discard = e2e >= t_i
        q = np.where(discard, 0.0, (1.0 - e2e / t_i) * a_x)
        per_trial = q.mean(axis=1)
        out.append({
            "mu": float(mu), "mean_Q": float(per_trial.mean()), "std_Q": float(per_trial.std(ddof=1)) if trials > 1 else 0.0,
            "ci95_Q": float(1.96 * per_trial.std(ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0,
            "discard_rate": float(discard.mean()),
            "mean_e2e_ms": float(np.mean(np.where(np.isfinite(e2e), e2e, np.nan))) if np.isfinite(e2e).any() else float("inf"),
            "P_x": p_x, "mean_service_time_ms": 1000.0 / float(mu), "accuracy": float(rows["correct"].mean()),
        })
    return out


def write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["strategy"] + [k for k in results[0] if k != "strategy"])
        w.writeheader()
        w.writerows(results)


# Categorical slots 1-5 of the reference palette; colour follows the strategy, not file order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STRATEGY_COLORS = dict(zip(["HCCLIO", "DCI (conf. only)", "Edge only", "Cloud only", "Local only"], COLORS))


def plot(per_strategy: dict[str, list[dict]], tier: str, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for i, (label, res) in enumerate(per_strategy.items()):
        mu = [r["mu"] for r in res]
        q = np.array([r["mean_Q"] for r in res])
        ci = np.array([r["ci95_Q"] for r in res])
        c = STRATEGY_COLORS.get(label, "#898781")
        ax.plot(mu, q, color=c, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(mu, q - ci, q + ci, color=c, alpha=0.15, lw=0)
    ax.set_xlabel(rf"$\mu_{{{tier}}}$ (frames/s)")
    ax.set_ylabel("mean QoE  $Q_x$")
    ax.grid(True, color="#e4e3dc", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
