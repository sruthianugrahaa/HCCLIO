"""main.py - Cloud server tier (Windows laptop): HCCLIO Algorithm 1, Cloud side.

Start-up
  1. config.py          load the shared common/hcclio.yaml
  2. model.py           load ViT-Large/16 once
  3. backhaul_delay.py  Gamma Edge<->Cloud backhaul delay model
  4. e2lm_server.py     E2LM probe server on TCP :9000
  5. mqtt_client.py     subscribe to hcclio/cloud/request on the broker (Mosquitto on the Edge)

Per frame (cascade from the Edge with p_i + p_ES, or direct from the IoT device with p_i)
  6. backhaul_delay.py  draw T_bh for this frame -> backhaul_delay_ms
  7. inference.py       p_CS = ViT-Large(JPEG), weighted ensemble, gate at 0.80
                          pass -> tier "Cloud"         result y_cloud
                          fail -> tier "Fallback-IoT"  result y_i
  8. publish the answer + all timings to the IoT device's reply topic

    python outputs/cloud_server_tier/main.py
    python outputs/cloud_server_tier/main.py --backend stub
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))  # outputs/ (holds common/)

from backhaul_delay import BackhaulDelay  # noqa: E402
from config import load  # noqa: E402
from e2lm_server import E2LMServer  # noqa: E402
from inference import cloud_inference  # noqa: E402
from model import ViTLarge  # noqa: E402
from mqtt_client import MqttClient  # noqa: E402

from common.messages import topics  # noqa: E402

log = logging.getLogger("cloud")


class CloudTier:
    def __init__(self, cfg, start_e2lm: bool = True):
        self.cfg = cfg
        self.model = ViTLarge(cfg.model_name, cfg.backend, cfg.accuracy)
        self.infer_lock = threading.Lock()  # one model, one frame at a time
        self.backhaul = BackhaulDelay(cfg.backhaul, cfg.seed + 1)
        self.backhaul_lock = threading.Lock()
        self.e2lm_server = E2LMServer(cfg.e2lm_port, cfg.e2lm_work_iters).start_background() if start_e2lm else None
        self.mqtt = MqttClient(cfg.broker_host, cfg.broker_port, "hcclio-cloud", cfg.qos)
        self.mqtt.subscribe(topics(cfg.topic_prefix)["cloud_request"], self.on_request)

    def on_request(self, msg: dict) -> None:
        try:
            self.handle(msg)
        except Exception:
            log.exception("frame %s failed", msg.get("frame_id"))

    def handle(self, msg: dict) -> None:
        # every frame that reaches the Cloud crossed the wired Edge -> Cloud backhaul
        with self.backhaul_lock:
            msg["timings"]["backhaul_delay_ms"] = self.backhaul.delay_ms()
        with self.infer_lock:
            r = cloud_inference(self.model, msg, self.cfg)
        msg["timings"]["cloud_inference_ms"] = r.cloud_inference_ms
        self.mqtt.publish(msg["reply_topic"], {
            "frame_id": msg["frame_id"], "tier": r.tier, "prediction_idx": r.prediction,
            "aggregated_conf": r.c_cloud, "route": msg["route"] + r.gate, "path": msg["path"] + "->Cloud",
            "timings": msg["timings"]})
        log.info("frame %4d  %-12s c=%.3f  backhaul=%.1f ms", msg["frame_id"], r.tier, r.c_cloud,
                 msg["timings"]["backhaul_delay_ms"])

    def close(self):
        self.mqtt.close()
        if self.e2lm_server:
            self.e2lm_server.shutdown()
            self.e2lm_server.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["timm", "stub"], default=None)
    ap.add_argument("--no-e2lm", action="store_true", help="do not start the E2LM probe server")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load(args.config, args.backend)
    cloud = CloudTier(cfg, start_e2lm=not args.no_e2lm)
    log.info("Cloud ready: %s, broker %s:%d, E2LM :%d", cfg.model_name, cfg.broker_host, cfg.broker_port,
             cfg.e2lm_port)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(1.0):
        pass
    cloud.close()


if __name__ == "__main__":
    main()
