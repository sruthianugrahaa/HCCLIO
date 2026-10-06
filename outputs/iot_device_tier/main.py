"""main.py - IoT device tier (Raspberry Pi 5): HCCLIO Algorithm 1, IoT side.

For each frame x:
  1. inference.py      ViT-Small  -> p_i, c_i, y_i
  2. confidence gate   c_i >= 0.80  -> answer locally (tier "IoT")
  3. otherwise         e2lm.py               -> delta_E2LM_edge (measured)
                       edge_rayleigh_delay.py -> delta_wl        (simulated)
  4. latency gate      delta_wl + delta_E2LM_edge <= 500 ms
                         yes -> offload JPEG + p_i to the Edge over MQTT
                                (Edge ensembles, may cascade to the Cloud)
                         no  -> offload straight to the Cloud
                                (+ delta_E2LM_cloud, + Gamma backhaul delay)
  5. the answer returns over MQTT; qoe.py computes T_x^E2E and Q_x;
     results.py writes the row to outputs/logs/qoe_coclio.csv

Start the Edge (run_edge.py) and Cloud (run_cloud.py) servers first.

    python outputs/iot_device_tier/main.py                 # all 1000 frames
    python outputs/iot_device_tier/main.py --limit 50      # quick check
    python outputs/iot_device_tier/main.py --backend stub  # no torch, simulated ViTs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

from e2lm import E2LM  # noqa: E402
from edge_rayleigh_delay import EdgeRayleighDelay  # noqa: E402
from inference import ViTSmall  # noqa: E402
from load_dataset import load_dataset  # noqa: E402
from qoe import e2e_latency_ms, qoe  # noqa: E402
from results import ResultsCSV, summarise  # noqa: E402

from hcclio.channel import GammaBackhaul  # noqa: E402
from hcclio.classes import class_name  # noqa: E402
from hcclio.config import load_config, resolve_path  # noqa: E402
from hcclio.transport import MqttTransport, ResponseWaiter, topics  # noqa: E402

log = logging.getLogger("iot")


class IoTTier:
    def __init__(self, cfg, backend: str):
        self.cfg = cfg
        self.tau_conf = float(cfg["gates"]["tau_conf"])          # 0.80
        self.tau_lat = float(cfg["gates"]["tau_lat_ms"])         # 500 ms
        self.acc = {"IoT": cfg["models"]["iot"]["accuracy"], "Fallback-IoT": cfg["models"]["iot"]["accuracy"],
                    "Edge": cfg["models"]["edge"]["accuracy"], "Cloud": cfg["models"]["cloud"]["accuracy"]}
        seed = int(cfg["sim"]["seed"])
        self.vit = ViTSmall(cfg["models"]["iot"]["name"], backend)
        self.wireless = EdgeRayleighDelay(cfg["wireless"], seed)
        self.backhaul = GammaBackhaul(cfg["backhaul"], np.random.default_rng(seed + 1))
        self.e2lm = E2LM(cfg["network"])
        self.t_i_ms = float(cfg["qoe"]["t_i_ms"])

        m = cfg["network"]["mqtt"]
        self.timeout = float(m["response_timeout_s"])
        t = topics(m["topic_prefix"])
        self.topic_edge, self.topic_cloud = t["edge_request"], t["cloud_request"]
        self.reply_topic = t["iot_response"].format(client="pi5-hcclio")
        self.mqtt = MqttTransport(m["broker_host"], int(m["broker_port"]), int(m["keepalive_s"]), int(m["qos"]),
                                  client_id="iot-pi5-hcclio")
        self.waiter = ResponseWaiter()
        self.mqtt.subscribe(self.reply_topic, self.waiter.on_message)

    def calibrate_t_i(self, frames) -> None:
        q = self.cfg["qoe"]
        if q["t_i_mode"] == "calibrated":
            times = [self.vit.infer(f.jpeg).t_ms for f in frames[: int(q["calib_frames"])]]
            self.t_i_ms = float(q["calib_scale"]) * float(np.mean(times))
        log.info("T_i^comp = %.1f ms (%s)", self.t_i_ms, q["t_i_mode"])

    def offload(self, topic, frame, payload) -> dict | None:
        msg = {"frame_id": frame.frame_id, "reply_topic": self.reply_topic, "strategy": "hcclio",
               "jpeg": frame.jpeg, **payload}
        self.waiter.expect(frame.frame_id)
        self.mqtt.publish(topic, msg)
        return self.waiter.wait(frame.frame_id, self.timeout)

    def process(self, frame) -> dict:
        # 1. local inference
        local = self.vit.infer(frame.jpeg)
        timings = {"local_inference_ms": local.t_ms}

        # 2. confidence gate
        if local.c_i >= self.tau_conf:
            answer = {"tier": "IoT", "prediction_idx": local.y_i, "aggregated_conf": local.c_i,
                      "route": "iot_conf_pass", "path": "IoT"}
        else:
            # 3. delays for the latency gate
            timings["E2LM_edge_ms"] = self.e2lm.edge_ms()
            timings["wireless_delay_ms"] = self.wireless.delay_ms()
            # 4. latency gate
            if timings["wireless_delay_ms"] + timings["E2LM_edge_ms"] <= self.tau_lat:
                route, topic, extra = "iot_conf_fail|lat_pass", self.topic_edge, {}
            else:
                timings["E2LM_cloud_ms"] = self.e2lm.cloud_ms()
                timings["backhaul_delay_ms"] = self.backhaul.delay_ms()
                route, topic, extra = "iot_conf_fail|lat_fail", self.topic_cloud, {"mode": "direct"}
            answer = self.offload(topic, frame, {"p_i": local.p_i, "timings": timings, "route": route,
                                                 "path": "IoT", **extra})
            if answer is None:  # no reply in time: keep the local answer, mark it
                answer = {"tier": "Timeout", "prediction_idx": local.y_i, "aggregated_conf": local.c_i,
                          "route": route + "|timeout", "path": "IoT", "timings": timings}
            timings = answer["timings"]

        # 5. E2E latency, QoE, CSV row
        tier = answer["tier"]
        e2e = e2e_latency_ms(timings)
        a_x = float(self.acc.get(tier, 0.0))
        q, discarded = qoe(e2e, self.t_i_ms, a_x)
        pred = int(answer["prediction_idx"])
        row = {
            "frame_id": frame.frame_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_file": frame.image_file, "ground_truth_idx": frame.ground_truth_idx,
            "ground_truth_name": frame.ground_truth_name,
            "tier": "Discard" if discarded else tier, "route": answer["route"], "path": answer["path"],
            "prediction_idx": pred, "class_name": class_name(pred), "correct": int(pred == frame.ground_truth_idx),
            "iot_confidence": local.c_i, "aggregated_conf": float(answer["aggregated_conf"]),
            "e2e_latency_ms": e2e, "A_x": a_x, "Q_x": q,
            "tau_conf": self.tau_conf, "tau_lat_ms": self.tau_lat, "T_i_ms": self.t_i_ms,
        }
        row.update({k: float(v) for k, v in timings.items()})
        return row

    def close(self):
        self.mqtt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="default: config/hcclio.yaml")
    ap.add_argument("--backend", choices=["timm", "stub"], default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", default=None, help="default: outputs/logs/qoe_coclio.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    backend = args.backend or cfg["models"]["backend"]
    frames = load_dataset(args.dataset or cfg["dataset"]["root"], args.limit or int(cfg["dataset"]["n_frames"]))
    out = Path(args.csv) if args.csv else resolve_path(cfg["logging"]["log_dir"]) / cfg["logging"]["hcclio_csv"]

    iot = IoTTier(cfg, backend)
    iot.calibrate_t_i(frames)
    t0 = time.time()
    with ResultsCSV(out) as results:
        for f in frames:
            row = iot.process(f)
            results.save(row)
            log.info("frame %4d  c_i=%.3f  %-12s %-38s e2e=%7.1f ms  Q=%.3f", f.frame_id, row["iot_confidence"],
                     row["tier"], row["route"], row["e2e_latency_ms"], row["Q_x"])
    iot.close()
    log.info("done in %.1f s -> %s", time.time() - t0, out)
    print(summarise(out))


if __name__ == "__main__":
    main()
