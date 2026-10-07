"""Benchmark: every frame -> Edge, no local ViT, no gates. Run on the IoT device.

Same as:  python outputs/iot_device_tier/main.py --strategy edge_only [--limit N]
Writes outputs/logs/qoe_edge_only.csv.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[1:1] = ["--strategy", "edge_only"]
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "iot_device_tier" / "main.py"), run_name="__main__")
