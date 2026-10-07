"""Dry run of the whole testbed on ONE computer (development only, not deployed).

Starts the Edge and Cloud tier main.py with simulated ViTs, runs the IoT tier
main.py for every strategy over synthetic frames, then make_report.py and
run_mu_sweep.py. Needs a local mosquitto broker (default 127.0.0.1:1883).

    python scripts/loopback.py --frames 100 --out /tmp/hcclio_dry
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STRATEGIES = ["hcclio", "local_only", "edge_only", "cloud_only", "dci"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"nothing listening on {port}")


def write_config(path: Path, broker_port: int, log_dir: Path, overrides: dict | None = None) -> Path:
    sys.path.insert(0, str(OUT))
    from common.settings import deep_update

    cfg = yaml.safe_load(open(OUT / "common" / "hcclio.yaml"))
    cfg = deep_update(cfg, {
        "models": {"backend": "stub"},
        "network": {"edge_host": "127.0.0.1", "cloud_host": "127.0.0.1",
                    "mqtt": {"broker_host": "127.0.0.1", "broker_port": broker_port, "response_timeout_s": 30},
                    "e2lm": {"edge_port": free_port(), "cloud_port": free_port(), "server_work_iters": 0}},
        "logging": {"log_dir": str(log_dir / "logs"), "report_dir": str(log_dir / "reports")},
    })
    if overrides:
        cfg = deep_update(cfg, overrides)
    path.write_text(yaml.safe_dump(cfg))
    return path


def run(out_dir: Path, frames: int = 60, broker_port: int = 1883, overrides: dict | None = None,
        strategies=STRATEGIES, analysis: bool = True, trace: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = write_config(out_dir / "loopback.yaml", broker_port, out_dir, overrides)
    cfg = yaml.safe_load(cfg_path.read_text())
    ds = out_dir / "dataset"
    py = sys.executable
    subprocess.run([py, str(OUT / "iot_device_tier" / "download_dataset.py"), "--source", "synthetic",
                    "-n", str(frames), "--out", str(ds)], check=True, stdout=subprocess.DEVNULL)
    servers = [subprocess.Popen([py, str(OUT / t / "main.py"), "--config", str(cfg_path)],
                                stdout=open(out_dir / f"{t}.log", "w"), stderr=subprocess.STDOUT)
               for t in ("edge_server_tier", "cloud_server_tier")]
    try:
        wait_port(cfg["network"]["e2lm"]["edge_port"])
        wait_port(cfg["network"]["e2lm"]["cloud_port"])
        time.sleep(0.5)  # MQTT subscriptions
        for s in strategies:
            subprocess.run([py, str(OUT / "iot_device_tier" / "main.py"), "--config", str(cfg_path),
                            "--strategy", s, "--dataset", str(ds)], check=True,
                           stdout=subprocess.DEVNULL, stderr=open(out_dir / f"iot_{s}.log", "w"))
        if trace:
            subprocess.run([py, str(OUT / "iot_device_tier" / "trace.py"), "--config", str(cfg_path),
                            "--dataset", str(ds)], check=True,
                           stdout=subprocess.DEVNULL, stderr=open(out_dir / "iot_trace.log", "w"))
    finally:
        for p in servers:
            p.terminate()
            p.wait()
    if analysis:
        subprocess.run([py, str(OUT / "make_report.py"), str(cfg_path)], check=True)
        for tier in ("i", "ES", "CS"):
            subprocess.run([py, str(OUT / "run_mu_sweep.py"), "--config", str(cfg_path), "--tier", tier,
                            "--trials", "100"], check=True, stdout=subprocess.DEVNULL)
        if trace:
            subprocess.run([py, str(OUT / "make_plot_data.py"), "--config", str(cfg_path), "--trials", "20"],
                           check=True, stdout=subprocess.DEVNULL)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("results in", run(Path(a.out or tempfile.mkdtemp(prefix="hcclio_dry_")), a.frames, a.broker_port))
