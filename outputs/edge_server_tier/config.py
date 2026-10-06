"""config.py - Edge tier settings, read from the shared config/hcclio.yaml.

Every tier reads the same YAML file so the thresholds, weights and network
addresses can never disagree between the Pi, the Edge and the laptop.

    python config.py            # print the Edge settings
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcclio.config import load_config  # noqa: E402


@dataclass
class EdgeConfig:
    # MQTT (Mosquitto runs on this machine)
    broker_host: str
    broker_port: int
    qos: int
    topic_prefix: str
    # E2LM probe server on this machine, and the Cloud's probe server
    e2lm_port: int
    e2lm_work_iters: int
    cloud_host: str
    cloud_e2lm_port: int
    e2lm_n_probes: int
    e2lm_timeout_s: float
    # model and gates
    model_name: str
    backend: str           # timm | stub
    accuracy: float        # A_ES
    tau_conf: float        # 0.80
    w_i: float             # 0.44
    w_ES: float            # 0.56
    dci_use_ensemble: bool
    # simulated channels (sampled here, sent to the IoT device)
    wireless: dict
    backhaul: dict
    seed: int
    raw: dict              # the full YAML, for anything else


def load(path: str | None = None, backend: str | None = None) -> EdgeConfig:
    c = load_config(path)
    n, e, m = c["network"], c["network"]["e2lm"], c["network"]["mqtt"]
    return EdgeConfig(
        broker_host=m["broker_host"], broker_port=int(m["broker_port"]), qos=int(m["qos"]),
        topic_prefix=m["topic_prefix"],
        e2lm_port=int(e.get("edge_port") or e["port"]), e2lm_work_iters=int(e.get("server_work_iters", 0)),
        cloud_host=n["cloud_host"], cloud_e2lm_port=int(e.get("cloud_port") or e["port"]),
        e2lm_n_probes=int(e.get("n_probes", 3)), e2lm_timeout_s=float(e.get("timeout_s", 2.0)),
        model_name=c["models"]["edge"]["name"], backend=backend or c["models"]["backend"],
        accuracy=float(c["models"]["edge"]["accuracy"]), tau_conf=float(c["gates"]["tau_conf"]),
        w_i=float(c["weights"]["edge"]["w_i"]), w_ES=float(c["weights"]["edge"]["w_ES"]),
        dci_use_ensemble=bool(c["dci"]["use_ensemble"]),
        wireless=dict(c["wireless"]), backhaul=dict(c["backhaul"]), seed=int(c["sim"]["seed"]), raw=c,
    )


if __name__ == "__main__":
    for k, v in vars(load(sys.argv[1] if len(sys.argv) > 1 else None)).items():
        if k != "raw":
            print(f"{k:18s} {v}")
