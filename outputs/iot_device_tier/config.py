"""config.py - IoT tier settings, read from the shared common/hcclio.yaml.

Every tier reads the same YAML file so the thresholds, weights and network
addresses can never disagree between the Pi, the Edge and the laptop.

    python config.py            # print the IoT settings
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # outputs/ (holds common/)
sys.path.insert(0, str(ROOT))

from common.settings import load_settings, resolve_path  # noqa: E402


@dataclass
class IoTConfig:
    # network
    broker_host: str           # Mosquitto on the Edge (10.0.17.25)
    broker_port: int
    qos: int
    topic_prefix: str
    response_timeout_s: float
    edge_host: str
    cloud_host: str
    edge_e2lm_port: int
    cloud_e2lm_port: int
    e2lm_n_probes: int
    e2lm_probe_bytes: int
    e2lm_timeout_s: float
    # model + gates
    model_name: str            # ViT-Small/16
    backend: str               # timm | stub
    tau_conf: float            # 0.80
    tau_lat_ms: float          # 500
    accuracy: dict             # A_x per answering tier
    # QoE
    e2e_mode: str              # measured (IoT-device stopwatch) | sum
    t_i_mode: str
    t_i_ms: float
    calib_frames: int
    calib_scale: float
    # files
    dataset_root: str
    n_frames: int
    csv_path: Path
    raw: dict                  # the full YAML, for anything else


def load(path: str | None = None, backend: str | None = None) -> IoTConfig:
    c = load_settings(path)
    n, e, m, q, mod = c["network"], c["network"]["e2lm"], c["network"]["mqtt"], c["qoe"], c["models"]
    return IoTConfig(
        broker_host=m["broker_host"], broker_port=int(m["broker_port"]), qos=int(m["qos"]),
        topic_prefix=m["topic_prefix"], response_timeout_s=float(m["response_timeout_s"]),
        edge_host=n["edge_host"], cloud_host=n["cloud_host"],
        edge_e2lm_port=int(e.get("edge_port") or e["port"]), cloud_e2lm_port=int(e.get("cloud_port") or e["port"]),
        e2lm_n_probes=int(e.get("n_probes", 3)), e2lm_probe_bytes=int(e.get("probe_bytes", 1024)),
        e2lm_timeout_s=float(e.get("timeout_s", 2.0)),
        model_name=mod["iot"]["name"], backend=backend or mod["backend"],
        tau_conf=float(c["gates"]["tau_conf"]), tau_lat_ms=float(c["gates"]["tau_lat_ms"]),
        accuracy={"IoT": float(mod["iot"]["accuracy"]), "Fallback-IoT": float(mod["iot"]["accuracy"]),
                  "Edge": float(mod["edge"]["accuracy"]), "Cloud": float(mod["cloud"]["accuracy"])},
        e2e_mode=q.get("e2e_mode", "measured"), t_i_mode=q["t_i_mode"], t_i_ms=float(q["t_i_ms"]), calib_frames=int(q["calib_frames"]),
        calib_scale=float(q["calib_scale"]),
        dataset_root=c["dataset"]["root"], n_frames=int(c["dataset"]["n_frames"]),
        csv_path=resolve_path(c["logging"]["log_dir"]) / c["logging"]["hcclio_csv"], raw=c,
    )


if __name__ == "__main__":
    for k, v in vars(load(sys.argv[1] if len(sys.argv) > 1 else None)).items():
        if k != "raw":
            print(f"{k:20s} {v}")
