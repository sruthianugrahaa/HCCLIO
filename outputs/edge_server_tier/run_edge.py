"""Edge server tier (Ubuntu, 10.0.17.25): ViT-Base + E2LM probe server on :9000.

    python outputs/edge_server_tier/run_edge.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

from hcclio.cli import edge_main  # noqa: E402

if __name__ == "__main__":
    edge_main()
