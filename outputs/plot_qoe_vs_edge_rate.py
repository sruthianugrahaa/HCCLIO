"""plot_qoe_vs_edge_rate.py - laptop: Average QoE vs Edge processing rate, HCCLIO vs benchmarks.

Plots reports/plot_data/qoe_vs_mu_edge.csv, written by make_plot_data.py. That file
replays every strategy at each Edge rate mu_ES (0.01-20 tasks/s by default) with the
Edge as an M/M/1 queue: each frame is re-decided at each rate, so HCCLIO's latency
gate moves frames off a slow Edge while CPO and Edge only keep sending to it.

    python outputs/make_plot_data.py
    python outputs/plot_qoe_vs_edge_rate.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STRATEGIES = [  # (name in the CSV, legend label, matplotlib style)
    ("HCCLIO", "HCCLIO (proposed)", dict(color="tab:red", marker="o", linewidth=2.2)),
    ("CPO", "CPO", dict(color="tab:blue", marker="s")),
    ("Edge only", "Edge only", dict(color="tab:green", marker="^")),
    ("Cloud only", "Cloud only", dict(color="tab:purple", marker="v", linestyle="--")),
    ("Local only", "Local only", dict(color="tab:gray", marker="D")),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "reports" / "plot_data" / "qoe_vs_mu_edge.csv"))
    ap.add_argument("--out", default=None, help="output name without extension (default: next to the CSV)")
    ap.add_argument("--linear", action="store_true", help="linear x axis (default: log)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path(args.csv)
    if not src.exists():
        raise SystemExit(f"{src} not found: run make_plot_data.py first")
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = Path(args.out) if args.out else src.with_name("qoe_vs_edge_rate")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, label, style in STRATEGIES:
        r = sorted((x for x in rows if x["strategy"] == name), key=lambda x: float(x["mu_ES"]))
        if not r:
            continue
        mu = np.array([float(x["mu_ES"]) for x in r])
        q = np.array([float(x["mean_QoE"]) for x in r])
        ci = np.array([float(x["ci95_QoE"]) for x in r])
        ax.plot(mu, q, label=label, markersize=5, **style)
        ax.fill_between(mu, q - ci, q + ci, color=style["color"], alpha=0.12)
        print(f"{label:18s} " + "  ".join(f"{m:.3g}:{v:.3f}" for m, v in zip(mu, q)))

    if not args.linear:
        ax.set_xscale("log")
    ax.set_xlabel(r"Edge processing rate $\mu_{ES}$ (tasks/s)")
    ax.set_ylabel("Average QoE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"saved {out}.png / .pdf")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
