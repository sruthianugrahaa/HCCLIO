"""IoT device tier (Raspberry Pi 5): run HCCLIO (Algorithm 1) over the dataset.

    python outputs/iot_device_tier/run_iot.py [--limit N] [--strategy hcclio]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from hcclio.cli import iot_main  # noqa: E402

if __name__ == "__main__":
    iot_main()
