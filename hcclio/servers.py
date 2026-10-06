"""Edge (ViT-Base) and Cloud (ViT-Large) tier servers.

Both subscribe over MQTT, run their model, form the weighted collaborative
ensemble, apply the confidence gate and either answer the IoT device directly
or (Edge only) cascade the frame to the Cloud.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .channel import GammaBackhaul, RayleighChannel
from .e2lm import E2LMClient, E2LMServer
from .models import build_classifier
from .qoe import ensemble, top1
from .transport import Transport, topics

log = logging.getLogger("hcclio")


class _TierServer:
    tier = ""
    request_key = ""

    def __init__(self, cfg, transport: Transport, classifier=None, start_e2lm: bool = False,
                 e2lm_port: int | None = None):
        self.cfg = cfg
        self.tx = transport
        self.topics = topics(cfg["network"]["mqtt"]["topic_prefix"])
        self.tau = float(cfg["gates"]["tau_conf"])
        self.model = classifier or build_classifier(cfg, self.tier)
        # one model instance, so serialise inference (torch is not re-entrant on CPU threads anyway)
        self._infer_lock = threading.Lock()
        self.e2lm_server = None
        if start_e2lm:
            e = cfg["network"]["e2lm"]
            port = e2lm_port or e.get(f"{self.tier}_port") or e["port"]
            self.e2lm_server = E2LMServer("0.0.0.0", int(port), int(e.get("server_work_iters", 0)),
                                          channel_fn=self.channel_sample)
            self.e2lm_server.start_background()
        self.tx.subscribe(self.topics[self.request_key], self._safe_handle)

    def channel_sample(self) -> dict:
        """Simulated delays this tier hands to the IoT device on an E2LM probe (none by default)."""
        return {}

    def _infer(self, jpeg: bytes) -> tuple[np.ndarray, float]:
        with self._infer_lock:
            return self.model.predict(jpeg)

    def _safe_handle(self, topic, msg):
        try:
            self.handle(msg)
        except Exception:  # keep the server alive; the IoT side times out and logs the frame
            log.exception("%s failed on frame %s", self.tier, msg.get("frame_id"))

    def respond(self, msg: dict, tier: str, pred: int, agg_conf: float, route: str) -> None:
        self.tx.publish(msg["reply_topic"], {
            "frame_id": msg["frame_id"], "tier": tier, "prediction_idx": int(pred),
            "aggregated_conf": float(agg_conf), "route": route, "path": msg["path"],
            "timings": msg["timings"],
        })

    def handle(self, msg: dict) -> None:
        raise NotImplementedError

    def close(self):
        if self.e2lm_server:
            self.e2lm_server.shutdown()
            self.e2lm_server.server_close()


class EdgeServer(_TierServer):
    tier = "edge"
    request_key = "edge_request"

    def __init__(self, cfg, transport, classifier=None, start_e2lm=False, rng=None, e2lm_port=None,
                 cloud_host: str | None = None):
        self.wireless = RayleighChannel(cfg["wireless"], rng or np.random.default_rng())
        super().__init__(cfg, transport, classifier, start_e2lm, e2lm_port)
        self.w = cfg["weights"]["edge"]
        self.e2lm = E2LMClient(cfg["network"]["e2lm"])
        self.cloud_host = cloud_host or cfg["network"]["cloud_host"]
        self.cloud_e2lm_port = cfg["network"]["e2lm"].get("cloud_port")
        self.dci_ensemble = bool(cfg["dci"]["use_ensemble"])

    def channel_sample(self) -> dict:
        """The Edge simulates the IoT<->Edge Rayleigh delay for one frame and hands it to the IoT device."""
        return {"wireless_delay_ms": self.wireless.delay_ms()}

    def handle(self, msg):
        p_es, t_es = self._infer(msg["jpeg"])
        msg["timings"]["edge_inference_ms"] = t_es
        msg["path"] = msg["path"] + "->Edge"
        strategy = msg["strategy"]

        if strategy == "edge_only" or msg.get("p_i") is None:
            c, y = top1(p_es)
            return self.respond(msg, "Edge", y, c, msg["route"])

        if strategy == "dci" and not self.dci_ensemble:
            c_edge, y_edge = top1(p_es)
        else:
            _, c_edge, y_edge = ensemble([msg["p_i"], p_es], [self.w["w_i"], self.w["w_ES"]])

        if c_edge >= self.tau:
            return self.respond(msg, "Edge", y_edge, c_edge, msg["route"] + "|edge_conf_pass")

        # cascade to Cloud with p_i, p_ES and the JPEG
        msg["timings"]["E2LM_cloud_ms"] = self.e2lm(self.cloud_host, self.cloud_e2lm_port)
        msg["route"] += "|edge_conf_fail"
        msg["p_es"] = p_es
        msg["mode"] = "cascade"
        msg["edge_conf"] = c_edge
        self.tx.publish(self.topics["cloud_request"], msg)


class CloudServer(_TierServer):
    tier = "cloud"
    request_key = "cloud_request"

    def __init__(self, cfg, transport, classifier=None, start_e2lm=False, e2lm_port=None, rng=None):
        super().__init__(cfg, transport, classifier, start_e2lm, e2lm_port)
        self.backhaul = GammaBackhaul(cfg["backhaul"], rng or np.random.default_rng())
        self.w = cfg["weights"]["cloud"]
        self.wd = cfg["weights"]["cloud_direct"]
        self.dci_ensemble = bool(cfg["dci"]["use_ensemble"])

    def handle(self, msg):
        # every frame that reaches the Cloud crossed the wired Edge -> Cloud backhaul
        msg["timings"]["backhaul_delay_ms"] = self.backhaul.delay_ms()
        p_cs, t_cs = self._infer(msg["jpeg"])
        msg["timings"]["cloud_inference_ms"] = t_cs
        msg["path"] = msg["path"] + "->Cloud"
        strategy = msg["strategy"]
        p_i = msg.get("p_i")

        if strategy == "cloud_only" or p_i is None:
            c, y = top1(p_cs)
            return self.respond(msg, "Cloud", y, c, msg["route"])

        mode = msg.get("mode", "direct")
        if strategy == "dci" and not self.dci_ensemble:
            c_cloud, y_cloud = top1(p_cs)
        elif mode == "cascade":
            _, c_cloud, y_cloud = ensemble([p_i, msg["p_es"], p_cs], [self.w["w_i"], self.w["w_ES"], self.w["w_CS"]])
        else:  # direct IoT -> Cloud, no Edge vector in the ensemble
            _, c_cloud, y_cloud = ensemble([p_i, p_cs], [self.wd["w_i"], self.wd["w_CS"]])

        if c_cloud >= self.tau:
            return self.respond(msg, "Cloud", y_cloud, c_cloud, msg["route"] + "|cloud_conf_pass")
        _, y_i = top1(p_i)
        self.respond(msg, "Fallback-IoT", y_i, c_cloud, msg["route"] + "|cloud_conf_fail")
