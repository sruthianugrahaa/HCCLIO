"""config.py - Cloud tier settings, read from the shared config/hcclio.yaml.

Every tier reads the same YAML file so the thresholds, weights and network
addresses can never disagree between the Pi, the Edge and the laptop.

    python config.py            # print the Cloud settings
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcclio.config import load_config  # noqa: E402


@dataclass
class CloudConfig:
    # MQTT (the broker is Mosquitto on the Edge machine)
    broker_host: str
    broker_port: int
    qos: int
    topic_prefix: str
    # E2LM probe server on this machine
    e2lm_port: int
    e2lm_work_iters: int
    # model and gates
    model_name: str
    backend: str               # timm | stub
    accuracy: float            # A_CS
    tau_conf: float            # 0.80
    w_cascade: dict            # w_i, w_ES, w_CS = 0.28 / 0.34 / 0.38
    w_direct: dict             # w_i, w_CS renormalised (no Edge vector)
    dci_use_ensemble: bool
    # simulated Edge <-> Cloud backhaul (sampled here)
    backhaul: dict
    seed: int
    raw: dict                  # the full YAML, for anything else


def load(path: str | None = None, backend: str | None = None) -> CloudConfig:
    c = load_config(path)
    e, m = c["network"]["e2lm"], c["network"]["mqtt"]
    return CloudConfig(
        broker_host=m["broker_host"], broker_port=int(m["broker_port"]), qos=int(m["qos"]),
        topic_prefix=m["topic_prefix"],
        e2lm_port=int(e.get("cloud_port") or e["port"]), e2lm_work_iters=int(e.get("server_work_iters", 0)),
        model_name=c["models"]["cloud"]["name"], backend=backend or c["models"]["backend"],
        accuracy=float(c["models"]["cloud"]["accuracy"]), tau_conf=float(c["gates"]["tau_conf"]),
        w_cascade=dict(c["weights"]["cloud"]), w_direct=dict(c["weights"]["cloud_direct"]),
        dci_use_ensemble=bool(c["dci"]["use_ensemble"]),
        backhaul=dict(c["backhaul"]), seed=int(c["sim"]["seed"]), raw=c,
    )


if __name__ == "__main__":
    for k, v in vars(load(sys.argv[1] if len(sys.argv) > 1 else None)).items():
        if k != "raw":
            print(f"{k:18s} {v}")
