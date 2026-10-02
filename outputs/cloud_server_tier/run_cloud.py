"""Cloud server tier (Windows laptop): ViT-Large + E2LM probe server on :9000.

    python outputs/cloud_server_tier/run_cloud.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from hcclio.cli import cloud_main  # noqa: E402

if __name__ == "__main__":
    cloud_main()
