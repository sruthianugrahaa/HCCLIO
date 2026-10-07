"""results.py - saves every frame's result as one row of outputs/logs/qoe_coclio.csv.

    python results.py outputs/logs/qoe_coclio.csv     # prints a short summary of a log
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

COLUMNS = [
    "frame_id", "timestamp", "image_file", "ground_truth_idx", "ground_truth_name",
    "tier", "route", "path", "prediction_idx", "class_name", "correct",
    "iot_confidence", "aggregated_conf",
    "local_inference_ms", "wireless_delay_ms", "E2LM_edge_ms", "edge_inference_ms",
    "backhaul_delay_ms", "E2LM_cloud_ms", "cloud_inference_ms",
    "e2e_latency_ms", "iot_wallclock_ms", "e2e_sum_ms", "A_x", "Q_x", "tau_conf", "tau_lat_ms", "T_i_ms",
]


class ResultsCSV:
    """Writes the header once, then one row per frame (flushed so a crash keeps every finished frame)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction="ignore")
        self._w.writeheader()

    def save(self, row: dict) -> None:
        self._w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def summarise(path) -> str:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    if not n:
        return "no rows"
    mean = lambda c: sum(float(r[c] or 0) for r in rows) / n
    tiers = Counter(r["tier"] for r in rows)
    shares = "  ".join(f"{t}={c / n:.1%}" for t, c in tiers.most_common())
    return (f"{n} frames  accuracy={mean('correct'):.3f}  mean Q={mean('Q_x'):.3f}  "
            f"mean e2e={mean('e2e_latency_ms'):.1f} ms\n{shares}")


if __name__ == "__main__":
    import sys

    print(summarise(sys.argv[1] if len(sys.argv) > 1 else "outputs/logs/qoe_coclio.csv"))
