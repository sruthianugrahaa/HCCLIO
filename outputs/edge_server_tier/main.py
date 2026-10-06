"""main.py - Edge server tier (Ubuntu 10.0.17.25): HCCLIO Algorithm 1, Edge side.

Start-up
  1. config.py                load the shared config/hcclio.yaml
  2. models.py                load ViT-Base/16 once
  3. edge_rayleigh_delay.py   Rayleigh IoT<->Edge channel, sampled per frame and sent to the
                              IoT device in the E2LM reply
  4. e2lm.py                  E2LM probe server on TCP :9000
  5. mqtt_client.py           subscribe to hcclio/edge/request on the Mosquitto broker

Per offloaded frame (message from the IoT device: JPEG, p_i, timings)
  6. inference.py   p_ES = ViT-Base(JPEG); p_ens = 0.44 p_i + 0.56 p_ES; gate at 0.80
       pass -> publish the answer to the IoT device's reply topic       (tier "Edge")
       fail -> probe the Cloud (delta_E2LM_cloud) and publish JPEG + p_i + p_ES
               to hcclio/cloud/request (cascade; the Cloud adds the backhaul delay)

Mosquitto must be running on this machine first (see README).

    python outputs/edge_server_tier/main.py
    python outputs/edge_server_tier/main.py --backend stub
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
sys.path.insert(1, str(HERE.parents[1]))

from config import load  # noqa: E402
from e2lm import E2LMServer, make_channel_fn, probe_ms  # noqa: E402
from edge_rayleigh_delay import EdgeRayleighDelay  # noqa: E402
from inference import edge_inference  # noqa: E402
from models import ViTBase  # noqa: E402
from mqtt_client import MqttClient  # noqa: E402

from hcclio.transport import topics  # noqa: E402

log = logging.getLogger("edge")


class EdgeTier:
    def __init__(self, cfg, start_e2lm: bool = True):
        self.cfg = cfg
        self.model = ViTBase(cfg.model_name, cfg.backend, cfg.accuracy)
        self.infer_lock = threading.Lock()  # one model, one frame at a time
        self.e2lm_server = None
        if start_e2lm:
            channel_fn = make_channel_fn(EdgeRayleighDelay(cfg.wireless, cfg.seed))
            self.e2lm_server = E2LMServer(cfg.e2lm_port, cfg.e2lm_work_iters, channel_fn).start_background()
        t = topics(cfg.topic_prefix)
        self.topic_cloud = t["cloud_request"]
        self.mqtt = MqttClient(cfg.broker_host, cfg.broker_port, "hcclio-edge", cfg.qos)
        self.mqtt.subscribe(t["edge_request"], self.on_request)

    def on_request(self, msg: dict) -> None:
        try:
            self.handle(msg)
        except Exception:
            log.exception("frame %s failed", msg.get("frame_id"))

    def handle(self, msg: dict) -> None:
        with self.infer_lock:
            r = edge_inference(self.model, msg["jpeg"], msg.get("p_i"), msg["strategy"], self.cfg)
        msg["timings"]["edge_inference_ms"] = r.edge_inference_ms
        msg["path"] += "->Edge"

        if r.answer_here:
            route = msg["route"] if msg.get("p_i") is None or msg["strategy"] == "edge_only" \
                else msg["route"] + "|edge_conf_pass"
            self.mqtt.publish(msg["reply_topic"], {
                "frame_id": msg["frame_id"], "tier": "Edge", "prediction_idx": r.y_edge,
                "aggregated_conf": r.c_edge, "route": route, "path": msg["path"], "timings": msg["timings"]})
            log.info("frame %4d  Edge  c=%.3f", msg["frame_id"], r.c_edge)
            return

        # cascade to the Cloud
        msg["timings"]["E2LM_cloud_ms"] = probe_ms(self.cfg.cloud_host, self.cfg.cloud_e2lm_port,
                                                   self.cfg.e2lm_n_probes, timeout_s=self.cfg.e2lm_timeout_s)
        msg.update({"route": msg["route"] + "|edge_conf_fail", "p_es": r.p_ES, "mode": "cascade",
                    "edge_conf": r.c_edge})
        self.mqtt.publish(self.topic_cloud, msg)
        log.info("frame %4d  -> Cloud  c=%.3f", msg["frame_id"], r.c_edge)

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
    edge = EdgeTier(cfg, start_e2lm=not args.no_e2lm)
    log.info("Edge ready: %s, broker %s:%d, E2LM :%d", cfg.model_name, cfg.broker_host, cfg.broker_port,
             cfg.e2lm_port)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(1.0):
        pass
    edge.close()


if __name__ == "__main__":
    main()
