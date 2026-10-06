"""make_report.py - accuracy / QoE / latency / tier-share table for every strategy log.

Runs on the laptop (or any computer) after copying the Pi's outputs/logs/.

    python outputs/make_report.py   ->  outputs/reports/summary.csv and summary.md
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

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


def summarise(csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    tiers = Counter(r["tier"] for r in rows)
    col = lambda c: np.array([_f(r.get(c)) for r in rows])
    e2e = col("e2e_latency_ms")
    out = {
        "strategy": STRATEGY_LABELS.get(csv_path.name, csv_path.stem), "file": csv_path.name, "frames": n,
        "accuracy": float(col("correct").mean()) if n else 0.0,
        "mean_Q": float(col("Q_x").mean()) if n else 0.0,
        "mean_e2e_ms": float(e2e.mean()) if n else 0.0,
        "p95_e2e_ms": float(np.percentile(e2e, 95)) if n else 0.0,
    }
    for t in ("IoT", "Edge", "Cloud", "Fallback-IoT", "Discard", "Timeout"):
        out[f"share_{t}"] = tiers.get(t, 0) / n if n else 0.0
    return out


def make_report(log_dir: Path, report_dir: Path) -> list[dict]:
    files = sorted(Path(log_dir).glob("qoe_*.csv"))
    summaries = [summarise(f) for f in files]
    if not summaries:
        return []
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    cols = ["strategy", "frames", "accuracy", "mean_Q", "mean_e2e_ms", "p95_e2e_ms",
            "share_IoT", "share_Edge", "share_Cloud", "share_Fallback-IoT", "share_Discard"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summaries:
        lines.append("| " + " | ".join(f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols) + " |")
    (report_dir / "summary.md").write_text("# HCCLIO strategy summary\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return summaries


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # outputs/ (holds common/)
    from common.settings import load_settings, resolve_path

    cfg = load_settings(sys.argv[1] if len(sys.argv) > 1 else None)
    report_dir = resolve_path(cfg["logging"]["report_dir"])
    rows = make_report(resolve_path(cfg["logging"]["log_dir"]), report_dir)
    print((report_dir / "summary.md").read_text() if rows else "no logs found")
