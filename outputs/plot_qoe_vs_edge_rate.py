"""plot_qoe_vs_edge_rate.py - laptop: Average QoE vs Edge processing rate, HCCLIO vs benchmarks.

Reads the per-image files written by make_plot_data.py
(reports/plot_data/per_frame/per_frame_<strategy>.csv) and sweeps the Edge
processing rate mu_ES (tasks/s). For every image that went through the Edge,
the measured Edge time is replaced by an exponential draw with mean 1000/mu ms:

    T_new = e2e_latency_ms - edge_inference_ms + Exp(1000/mu)
    Q     = (1 - T_new / T_i_ms) * A_x,   0 if T_new >= T_i_ms   (Discard)

Images that never visit the Edge (Local only, Cloud only, frames HCCLIO
answered on the Pi or sent straight to the Cloud) keep their logged QoE.
Each mu is averaged over --trials Monte-Carlo draws.

    python outputs/plot_qoe_vs_edge_rate.py
    python outputs/plot_qoe_vs_edge_rate.py --mu-min 5 --mu-max 40 --step 1 --trials 50
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STRATEGIES = [  # (file suffix, legend label, matplotlib style)
    ("hcclio", "HCCLIO (proposed)", dict(color="tab:red", marker="o", linewidth=2.2)),
    ("cpo", "CPO", dict(color="tab:blue", marker="s")),
    ("edge_only", "Edge only", dict(color="tab:green", marker="^")),
    ("cloud_only", "Cloud only", dict(color="tab:purple", marker="v")),
    ("local_only", "Local only", dict(color="tab:gray", marker="D")),
]


def num(v: str) -> float:
    return float(v) if v not in ("", None) else np.nan


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = ["e2e_latency_ms", "edge_inference_ms", "A_x", "T_i_ms", "Q_x"]
    return {c: np.array([num(r[c]) for r in rows]) for c in cols}


def avg_qoe(d: dict[str, np.ndarray], mu: float, trials: int, rng: np.random.Generator) -> tuple[float, float]:
    """Mean QoE over all images, and its standard deviation across trials."""
    uses_edge = ~np.isnan(d["edge_inference_ms"])
    if not uses_edge.any():
        q = float(np.mean(d["Q_x"]))
        return q, 0.0
    base = d["e2e_latency_ms"] - np.nan_to_num(d["edge_inference_ms"])
    means = []
    for _ in range(trials):
        edge = rng.exponential(1000.0 / mu, size=len(base))
        t = np.where(uses_edge, base + edge, d["e2e_latency_ms"])
        q = np.where(t >= d["T_i_ms"], 0.0, (1.0 - t / d["T_i_ms"]) * d["A_x"])
        q = np.where(uses_edge, q, d["Q_x"])
        means.append(q.mean())
    return float(np.mean(means)), float(np.std(means))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE / "reports" / "plot_data" / "per_frame"),
                    help="folder with per_frame_<strategy>.csv")
    ap.add_argument("--mu-min", type=float, default=5)
    ap.add_argument("--mu-max", type=float, default=40)
    ap.add_argument("--step", type=float, default=5)
    ap.add_argument("--trials", type=int, default=50, help="Monte-Carlo draws per mu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="output name without extension (default: <dir>/../qoe_vs_edge_rate)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(args.dir)
    mus = np.arange(args.mu_min, args.mu_max + 1e-9, args.step)
    out = Path(args.out) if args.out else folder.parent / "qoe_vs_edge_rate"
    rng = np.random.default_rng(args.seed)

    table = {"mu_ES_tasks_per_s": mus}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, style in STRATEGIES:
        path = folder / f"per_frame_{key}.csv"
        if not path.exists():
            print(f"skipped {label}: {path} not found")
            continue
        d = load(path)
        res = [avg_qoe(d, mu, args.trials, rng) for mu in mus]
        mean = np.array([r[0] for r in res])
        std = np.array([r[1] for r in res])
        table[f"{label}_avg_QoE"] = mean
        table[f"{label}_std"] = std
        ax.plot(mus, mean, label=label, markersize=5, **style)
        ax.fill_between(mus, mean - std, mean + std, color=style["color"], alpha=0.12)
        print(f"{label:18s} " + "  ".join(f"{m:g}:{q:.3f}" for m, q in zip(mus, mean)))

    ax.set_xlabel(r"Edge processing rate $\mu_{ES}$ (tasks/s)")
    ax.set_ylabel("Average QoE")
    ax.set_xlim(mus[0], mus[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(table.keys())
        for i in range(len(mus)):
            w.writerow([f"{table[k][i]:.6g}" for k in table])
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"saved {out}.png / .pdf / .csv")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
