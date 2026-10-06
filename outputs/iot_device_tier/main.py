"""main.py - IoT device tier (Raspberry Pi 5): HCCLIO Algorithm 1, IoT side.

For each frame x:
  1. load_dataset.py  next frame (JPEG + ground truth)
  2. inference.py     ViT-Small -> p_i, c_i, y_i
  3. confidence gate  c_i >= 0.80 -> answer locally (tier "IoT")
  4. otherwise        e2lm.py probes the Edge: measured delta_E2LM_edge, and the Edge
                      sends back the Rayleigh delay delta_wl it simulated for this
                      frame (and the backhaul delay). Both are stored for the frame.
  5. latency gate     delta_wl + delta_E2LM_edge <= 500 ms
                        yes -> mqtt_client.py: JPEG + p_i -> hcclio/edge/request
                               (Edge ensembles, may cascade to the Cloud)
                        no  -> e2lm.py probes the Cloud (delta_E2LM_cloud),
                               mqtt_client.py: JPEG + p_i -> hcclio/cloud/request
  6. the answer comes back on the IoT reply topic; qoe.py computes T_x^E2E and Q_x;
     results.py appends the row to outputs/logs/qoe_coclio.csv

Settings come from config.py (config/hcclio.yaml). Start the Edge
(edge_server_tier/main.py) and the Cloud (cloud_server_tier/run_cloud.py) first.

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
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))

from config import load  # noqa: E402
from e2lm import E2LM  # noqa: E402
from inference import ViTSmall  # noqa: E402
from load_dataset import load_dataset  # noqa: E402
from mqtt_client import MqttClient  # noqa: E402
from qoe import e2e_latency_ms, qoe  # noqa: E402
from results import ResultsCSV, summarise  # noqa: E402

from hcclio.classes import class_name  # noqa: E402
from hcclio.transport import topics  # noqa: E402

log = logging.getLogger("iot")


class IoTTier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vit = ViTSmall(cfg.model_name, cfg.backend)
        self.e2lm = E2LM(cfg)
        self.t_i_ms = cfg.t_i_ms
        t = topics(cfg.topic_prefix)
        self.topic_edge, self.topic_cloud = t["edge_request"], t["cloud_request"]
        self.mqtt = MqttClient(cfg.broker_host, cfg.broker_port, "iot-pi5-hcclio",
                               t["iot_response"].format(client="pi5-hcclio"), cfg.qos)

    def calibrate_t_i(self, frames) -> None:
        if self.cfg.t_i_mode == "calibrated":
            times = [self.vit.infer(f.jpeg).t_ms for f in frames[: self.cfg.calib_frames]]
            self.t_i_ms = self.cfg.calib_scale * float(np.mean(times))
        log.info("T_i^comp = %.1f ms (%s)", self.t_i_ms, self.cfg.t_i_mode)

    def process(self, frame) -> dict:
        cfg = self.cfg
        # 2. local inference
        local = self.vit.infer(frame.jpeg)
        timings = {"local_inference_ms": local.t_ms}

        # 3. confidence gate
        if local.c_i >= cfg.tau_conf:
            answer = {"tier": "IoT", "prediction_idx": local.y_i, "aggregated_conf": local.c_i,
                      "route": "iot_conf_pass", "path": "IoT"}
        else:
            # 4. E2LM to the Edge + the Edge's simulated channel delays for this frame
            timings["E2LM_edge_ms"], channel = self.e2lm.edge_probe()
            timings["wireless_delay_ms"] = float(channel.get("wireless_delay_ms", 0.0))
            backhaul_ms = float(channel.get("backhaul_delay_ms", 0.0))
            msg = {"frame_id": frame.frame_id, "strategy": "hcclio", "jpeg": frame.jpeg, "p_i": local.p_i,
                   "path": "IoT", "timings": timings, "backhaul_delay_ms": backhaul_ms}
            # 5. latency gate
            if timings["wireless_delay_ms"] + timings["E2LM_edge_ms"] <= cfg.tau_lat_ms:
                msg["route"] = "iot_conf_fail|lat_pass"
                answer = self.mqtt.request(self.topic_edge, msg, cfg.response_timeout_s)
            else:
                timings["E2LM_cloud_ms"] = self.e2lm.cloud_ms()
                timings["backhaul_delay_ms"] = backhaul_ms
                msg.update(route="iot_conf_fail|lat_fail", mode="direct")
                answer = self.mqtt.request(self.topic_cloud, msg, cfg.response_timeout_s)
            if answer is None:  # no reply in time: keep the local answer, mark it
                answer = {"tier": "Timeout", "prediction_idx": local.y_i, "aggregated_conf": local.c_i,
                          "route": msg["route"] + "|timeout", "path": "IoT", "timings": timings}
            timings = answer["timings"]

        # 6. E2E latency, QoE, CSV row
        tier = answer["tier"]
        e2e = e2e_latency_ms(timings)
        a_x = cfg.accuracy.get(tier, 0.0)
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
            "tau_conf": cfg.tau_conf, "tau_lat_ms": cfg.tau_lat_ms, "T_i_ms": self.t_i_ms,
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

    cfg = load(args.config, args.backend)
    frames = load_dataset(args.dataset or cfg.dataset_root, args.limit or cfg.n_frames)
    out = Path(args.csv) if args.csv else cfg.csv_path

    iot = IoTTier(cfg)
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
