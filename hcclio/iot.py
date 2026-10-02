"""IoT device tier (Raspberry Pi 5, ViT-Small) and the per-frame strategy logic.

Strategies
  hcclio      Algorithm 1: confidence gate + latency gate + weighted ensembles
  local_only  ViT-Small only, never offload
  edge_only   every frame -> Edge, no local ViT, no gates
  cloud_only  every frame -> Cloud, no local ViT
  dci         distributed benchmark (Zhang et al.): confidence gate only, no latency gate
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np

from .channel import GammaBackhaul, RayleighChannel
from .classes import class_name
from .dataset_loader import DatasetLoader, Frame
from .e2lm import E2LMClient
from .models import build_classifier
from .qoe import CsvLogger, e2e_latency, qoe, tier_accuracy, top1
from .transport import ResponseWaiter, Transport, topics

log = logging.getLogger("hcclio")

STRATEGIES = ("hcclio", "local_only", "edge_only", "cloud_only", "dci")


class IoTDevice:
    def __init__(self, cfg, transport: Transport | None, strategy: str = "hcclio", classifier=None,
                 client_id: str = "pi5", rng: np.random.Generator | None = None,
                 edge_host: str | None = None, cloud_host: str | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.cfg = cfg
        self.strategy = strategy
        self.tx = transport
        self.tau = float(cfg["gates"]["tau_conf"])
        self.tau_lat = float(cfg["gates"]["tau_lat_ms"])
        rng = rng or np.random.default_rng(cfg["sim"]["seed"])
        self.wireless = RayleighChannel(cfg["wireless"], rng)
        self.backhaul = GammaBackhaul(cfg["backhaul"], rng)
        e = cfg["network"]["e2lm"]
        self.e2lm = E2LMClient(e)
        self.edge_host = edge_host or cfg["network"]["edge_host"]
        self.cloud_host = cloud_host or cfg["network"]["cloud_host"]
        self.edge_e2lm_port = e.get("edge_port")
        self.cloud_e2lm_port = e.get("cloud_port")
        self.timeout = float(cfg["network"]["mqtt"]["response_timeout_s"])
        needs_local = strategy not in ("edge_only", "cloud_only")
        self.model = (classifier or build_classifier(cfg, "iot")) if needs_local else None
        self.t_i_ms = float(cfg["qoe"]["t_i_ms"])
        if transport is not None:
            t = topics(cfg["network"]["mqtt"]["topic_prefix"])
            self.t_edge, self.t_cloud = t["edge_request"], t["cloud_request"]
            self.reply_topic = t["iot_response"].format(client=f"{client_id}-{strategy}")
            self.waiter = ResponseWaiter()
            transport.subscribe(self.reply_topic, self.waiter.on_message)

    # ------------------------------------------------------------ helpers
    def calibrate_t_i(self, frames: list[Frame]) -> float:
        """t_i_mode: calibrated -> T_i^comp = calib_scale * mean ViT-Small time on the IoT device."""
        q = self.cfg["qoe"]
        if q["t_i_mode"] == "calibrated":
            model = self.model or build_classifier(self.cfg, "iot")
            times = [model.predict(f.jpeg)[1] for f in frames[: int(q["calib_frames"])]]
            self.t_i_ms = float(q["calib_scale"]) * float(np.mean(times))
            log.info("calibrated T_i^comp = %.1f ms", self.t_i_ms)
        return self.t_i_ms

    def _offload(self, topic: str, frame: Frame, payload: dict) -> dict | None:
        msg = {"frame_id": frame.frame_id, "reply_topic": self.reply_topic, "strategy": self.strategy,
               "jpeg": frame.jpeg, **payload}
        self.waiter.expect(frame.frame_id)
        self.tx.publish(topic, msg)
        return self.waiter.wait(frame.frame_id, self.timeout)

    # ------------------------------------------------------- one frame
    def process(self, frame: Frame) -> dict:
        timings: dict[str, float] = {}
        row = {"frame_id": frame.frame_id, "timestamp": datetime.now(timezone.utc).isoformat(),
               "image_file": frame.image_file, "ground_truth_idx": frame.ground_truth_idx,
               "ground_truth_name": frame.ground_truth_name,
               "tau_conf": self.tau, "tau_lat_ms": self.tau_lat}
        p_i = c_i = y_i = None
        if self.model is not None:
            p_i, timings["local_inference_ms"] = self.model.predict(frame.jpeg)
            c_i, y_i = top1(p_i)
            row["iot_confidence"] = c_i

        resp = None
        if self.strategy == "local_only" or (c_i is not None and c_i >= self.tau):
            route = "local_only" if self.strategy == "local_only" else "iot_conf_pass"
            resp = {"tier": "IoT", "prediction_idx": y_i, "aggregated_conf": c_i, "route": route, "path": "IoT"}
        else:
            timings["wireless_delay_ms"] = self.wireless.delay_ms()
            if self.strategy == "cloud_only":
                go_edge = False
                route = "cloud_only"
            elif self.strategy == "edge_only":
                timings["E2LM_edge_ms"] = self.e2lm(self.edge_host, self.edge_e2lm_port)
                go_edge, route = True, "edge_only"
            else:
                timings["E2LM_edge_ms"] = self.e2lm(self.edge_host, self.edge_e2lm_port)
                if self.strategy == "dci":
                    go_edge, route = True, "iot_conf_fail"
                else:
                    go_edge = timings["wireless_delay_ms"] + timings["E2LM_edge_ms"] <= self.tau_lat
                    route = "iot_conf_fail|lat_pass" if go_edge else "iot_conf_fail|lat_fail"
            payload = {"p_i": p_i, "timings": timings, "route": route, "path": "IoT"}
            if go_edge:
                resp = self._offload(self.t_edge, frame, payload)
            else:
                # straight to Cloud: wireless hop to the Edge AP, then the wired backhaul
                timings["E2LM_cloud_ms"] = self.e2lm(self.cloud_host, self.cloud_e2lm_port)
                timings["backhaul_delay_ms"] = self.backhaul.delay_ms()
                payload["mode"] = "direct"
                resp = self._offload(self.t_cloud, frame, payload)
            if resp is None:
                resp = {"tier": "Timeout", "prediction_idx": y_i if y_i is not None else -1,
                        "aggregated_conf": c_i or 0.0, "route": route + "|timeout", "path": "IoT", "timings": timings}
            timings = resp.get("timings", timings)

        for k, v in timings.items():
            row[k] = float(v)
        tier = resp["tier"]
        pred = int(resp["prediction_idx"])
        e2e = e2e_latency(timings)
        a_x = tier_accuracy(self.cfg, tier) if tier != "Timeout" else 0.0
        q, discarded = qoe(e2e, self.t_i_ms, a_x)
        row.update({
            "tier": "Discard" if discarded else tier,
            "route": resp["route"], "path": resp["path"],
            "prediction_idx": pred, "class_name": class_name(pred) if pred >= 0 else "",
            "correct": int(pred == frame.ground_truth_idx),
            "aggregated_conf": float(resp["aggregated_conf"]),
            "e2e_latency_ms": e2e, "A_x": a_x, "Q_x": q, "T_i_ms": self.t_i_ms,
        })
        return row

    # --------------------------------------------------------------- run
    def run(self, loader: DatasetLoader, csv_path, progress_every: int = 50) -> list[dict]:
        frames = list(loader)
        self.calibrate_t_i(frames)
        logger = CsvLogger(csv_path)
        rows = []
        t0 = time.time()
        try:
            for f in frames:
                row = self.process(f)
                logger.log(row)
                rows.append(row)
                if progress_every and (f.frame_id + 1) % progress_every == 0:
                    acc = np.mean([r["correct"] for r in rows])
                    qm = np.mean([r["Q_x"] for r in rows])
                    log.info("[%s] %d/%d frames  acc=%.3f  meanQ=%.3f  %.1fs", self.strategy, f.frame_id + 1,
                             len(frames), acc, qm, time.time() - t0)
        finally:
            logger.close()
        return rows
