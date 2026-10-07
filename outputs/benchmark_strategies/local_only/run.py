"""Benchmark: ViT-Small only, no offload. Run on the IoT device.

Same as:  python outputs/iot_device_tier/main.py --strategy local_only [--limit N]
Writes outputs/logs/qoe_local_only.csv.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[1:1] = ["--strategy", "local_only"]
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "iot_device_tier" / "main.py"), run_name="__main__")
