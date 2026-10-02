"""Benchmark: every frame -> Edge, no local ViT, no gates. Run on the IoT device.

    python outputs/benchmark_strategies/edge_only/run.py [--limit N]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

from hcclio.cli import iot_main  # noqa: E402

if __name__ == "__main__":
    iot_main(default_strategy="edge_only")
