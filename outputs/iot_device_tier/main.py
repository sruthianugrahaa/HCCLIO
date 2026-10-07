"""main.py - IoT device tier (Raspberry Pi 4): HCCLIO Algorithm 1, IoT side.

For each frame x:
  1. load_dataset.py  next frame (JPEG + ground truth)
  2. inference.py     ViT-Small -> p_i, c_i, y_i
  3. confidence gate  c_i >= 0.80 -> answer locally (tier "IoT")
  4. otherwise        e2lm.py probes the Edge: measured delta_E2LM_edge, and the Edge
                      sends back the Rayleigh delay delta_wl it simulated for this
                      frame. Both are stored for the frame.
  5. latency gate     delta_wl + delta_E2LM_edge <= 500 ms
                        yes -> mqtt_client.py: JPEG + p_i -> hcclio/edge/request
                               (Edge ensembles, may cascade to the Cloud)
                        no  -> e2lm.py probes the Cloud (delta_E2LM_cloud),
                               mqtt_client.py: JPEG + p_i -> hcclio/cloud/request
                               (the Cloud adds the backhaul delay)
  6. the answer comes back on the IoT reply topic; the IoT stopwatch gives T_x^E2E and qoe.py Q_x;
     results.py appends the row to outputs/logs/qoe_coclio.csv

Settings come from config.py (common/hcclio.yaml). Start the Edge
(edge_server_tier/main.py) and the Cloud (cloud_server_tier/run_cloud.py) first.

    python outputs/iot_device_tier/main.py                 # all 1000 frames
    python outputs/iot_device_tier/main.py --limit 50      # quick check
    python outputs/iot_device_tier/main.py --backend stub  # no torch, simulated ViTs
    python outputs/iot_device_tier/main.py --strategy edge_only   # a benchmark (see STRATEGIES)
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
sys.path.insert(1, str(HERE.parent))  # outputs/ (holds common/)

from config import load  # noqa: E402
from e2lm import E2LM  # noqa: E402
from inference import ViTSmall  # noqa: E402
from load_dataset import load_dataset  # noqa: E402
from mqtt_client import MqttClient  # noqa: E402
from qoe import e2e_latency_ms, iot_e2e_ms, qoe  # noqa: E402
from results import ResultsCSV, summarise  # noqa: E402

from common.classes import class_name  # noqa: E402
from common.messages import topics  # noqa: E402

log = logging.getLogger("iot")


STRATEGIES = {  # strategy -> CSV file in outputs/logs/
    "hcclio": "qoe_coclio.csv",
    "local_only": "qoe_local_only.csv",          # ViT-Small only, no offload
    "edge_only": "qoe_edge_only.csv",            # every frame -> Edge, no local ViT, no gates
    "cloud_only": "qoe_cloud_only.csv",          # every frame -> Cloud, no local ViT
    "dci": "qoe_distributed_benchmark.csv",      # DCI (Zhang et al.): confidence gate only
}


class IoTTier:
    def __init__(self, cfg, strategy: str = "hcclio"):
        self.cfg = cfg
        self.strategy = strategy
        self.vit = None if strategy in ("edge_only", "cloud_only") else ViTSmall(cfg.model_name, cfg.backend)
        self.e2lm = E2LM(cfg)
        self.t_i_ms = cfg.t_i_ms
        t = topics(cfg.topic_prefix)
        self.topic_edge, self.topic_cloud = t["edge_request"], t["cloud_request"]
        self.mqtt = None
        if strategy != "local_only":
            client = f"pi4-{strategy}"
            self.mqtt = MqttClient(cfg.broker_host, cfg.broker_port, f"iot-{client}",
                                   t["iot_response"].format(client=client), cfg.qos)

    def calibrate_t_i(self, frames) -> None:
        if self.cfg.t_i_mode == "calibrated":
            vit = self.vit or ViTSmall(self.cfg.model_name, self.cfg.backend)
            times = [vit.infer(f.jpeg).t_ms for f in frames[: self.cfg.calib_frames]]
            self.t_i_ms = self.cfg.calib_scale * float(np.mean(times))
        log.info("T_i^comp = %.1f ms (%s)", self.t_i_ms, self.cfg.t_i_mode)

    def process(self, frame) -> dict:
        cfg, strategy = self.cfg, self.strategy
        timings: dict = {}
        t0 = time.perf_counter()  # the IoT device's stopwatch: frame read -> answer received
        # 2. local inference (skipped by the edge_only / cloud_only benchmarks)
        local = self.vit.infer(frame.jpeg) if self.vit else None
        if local:
            timings["local_inference_ms"] = local.t_ms

        # 3. confidence gate
        if strategy == "local_only" or (local and local.c_i >= cfg.tau_conf):
            route = "local_only" if strategy == "local_only" else "iot_conf_pass"
            answer = {"tier": "IoT", "prediction_idx": local.y_i, "aggregated_conf": local.c_i,
                      "route": route, "path": "IoT"}
        else:
            # 4. E2LM to the Edge + the Rayleigh delay the Edge drew for this frame
            timings["E2LM_edge_ms"], channel = self.e2lm.edge_probe()
            timings["wireless_delay_ms"] = float(channel.get("wireless_delay_ms", 0.0))
            msg = {"frame_id": frame.frame_id, "strategy": strategy, "jpeg": frame.jpeg,
                   "p_i": local.p_i if local else None, "path": "IoT", "timings": timings}
            # 5. latency gate (HCCLIO only; DCI and edge_only always go to the Edge)
            if strategy == "cloud_only":
                go_edge, msg["route"] = False, "cloud_only"
                del timings["E2LM_edge_ms"]  # the frame only crosses the Edge AP, not its server
            elif strategy == "edge_only":
                go_edge, msg["route"] = True, "edge_only"
            elif strategy == "dci":
                go_edge, msg["route"] = True, "iot_conf_fail"
            else:
                go_edge = timings["wireless_delay_ms"] + timings["E2LM_edge_ms"] <= cfg.tau_lat_ms
                msg["route"] = "iot_conf_fail|lat_pass" if go_edge else "iot_conf_fail|lat_fail"
            if go_edge:
                answer = self.mqtt.request(self.topic_edge, msg, cfg.response_timeout_s)
            else:
                timings["E2LM_cloud_ms"] = self.e2lm.cloud_ms()
                msg["mode"] = "direct"
                answer = self.mqtt.request(self.topic_cloud, msg, cfg.response_timeout_s)
            if answer is None:  # no reply in time: keep the local answer (if any), mark it
                answer = {"tier": "Timeout", "prediction_idx": local.y_i if local else -1,
                          "aggregated_conf": local.c_i if local else 0.0,
                          "route": msg["route"] + "|timeout", "path": "IoT", "timings": timings}
            timings = answer["timings"]

        # 6. E2E latency as the IoT device experienced it, QoE, CSV row
        wallclock = (time.perf_counter() - t0) * 1000.0
        tier = answer["tier"]
        e2e_sum = e2e_latency_ms(timings)
        e2e_meas = iot_e2e_ms(wallclock, timings)
        e2e = e2e_meas if cfg.e2e_mode == "measured" else e2e_sum
        a_x = cfg.accuracy.get(tier, 0.0)
        q, discarded = qoe(e2e, self.t_i_ms, a_x)
        pred = int(answer["prediction_idx"])
        row = {
            "frame_id": frame.frame_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_file": frame.image_file, "ground_truth_idx": frame.ground_truth_idx,
            "ground_truth_name": frame.ground_truth_name,
            "tier": "Discard" if discarded else tier, "route": answer["route"], "path": answer["path"],
            "prediction_idx": pred, "class_name": class_name(pred) if pred >= 0 else "",
            "correct": int(pred == frame.ground_truth_idx),
            "iot_confidence": local.c_i if local else "", "aggregated_conf": float(answer["aggregated_conf"]),
            "e2e_latency_ms": e2e, "iot_wallclock_ms": wallclock, "e2e_sum_ms": e2e_sum,
            "A_x": a_x, "Q_x": q,
            "tau_conf": cfg.tau_conf, "tau_lat_ms": cfg.tau_lat_ms, "T_i_ms": self.t_i_ms,
        }
        row.update({k: float(v) for k, v in timings.items()})
        return row

    def close(self):
        if self.mqtt:
            self.mqtt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="default: common/hcclio.yaml")
    ap.add_argument("--backend", choices=["timm", "stub"], default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="hcclio",
                    help="hcclio (default) or one of the benchmarks")
    ap.add_argument("--csv", default=None, help="default: outputs/logs/<file for the strategy>")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load(args.config, args.backend)
    frames = load_dataset(args.dataset or cfg.dataset_root, args.limit or cfg.n_frames)
    out = Path(args.csv) if args.csv else cfg.csv_path.with_name(STRATEGIES[args.strategy])

    iot = IoTTier(cfg, args.strategy)
    iot.calibrate_t_i(frames)
    t0 = time.time()
    with ResultsCSV(out) as results:
        for f in frames:
            row = iot.process(f)
            results.save(row)
            log.info("frame %4d  %-12s %-38s e2e=%7.1f ms  Q=%.3f", f.frame_id,
                     row["tier"], row["route"], row["e2e_latency_ms"], row["Q_x"])
    iot.close()
    log.info("done in %.1f s -> %s", time.time() - t0, out)
    print(summarise(out))


if __name__ == "__main__":
    main()
