"""Benchmark: every frame -> Cloud, no local ViT. Run on the IoT device.

Same as:  python outputs/iot_device_tier/main.py --strategy cloud_only [--limit N]
Writes outputs/logs/qoe_cloud_only.csv.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[1:1] = ["--strategy", "cloud_only"]
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "iot_device_tier" / "main.py"), run_name="__main__")
