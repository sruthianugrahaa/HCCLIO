"""Kept for older instructions: same as `python outputs/cloud_server_tier/main.py`."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
