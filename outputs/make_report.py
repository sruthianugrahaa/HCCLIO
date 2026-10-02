"""Accuracy / QoE / latency / tier-share table for every strategy log.

    python outputs/make_report.py   ->  outputs/reports/summary.csv and summary.md
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root

from hcclio.config import load_config, resolve_path  # noqa: E402
from hcclio.report import make_report  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    report_dir = resolve_path(cfg["logging"]["report_dir"])
    rows = make_report(resolve_path(cfg["logging"]["log_dir"]), report_dir)
    print((report_dir / "summary.md").read_text() if rows else "no logs found")
