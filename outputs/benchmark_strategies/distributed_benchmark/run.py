"""Benchmark: DCI (Zhang et al.): confidence gate only, no latency gate. Run on the IoT device.

Same as:  python outputs/iot_device_tier/main.py --strategy dci [--limit N]
Writes outputs/logs/qoe_distributed_benchmark.csv.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[1:1] = ["--strategy", "dci"]
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "iot_device_tier" / "main.py"), run_name="__main__")
