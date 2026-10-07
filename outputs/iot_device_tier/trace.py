"""trace.py - IoT device tier: one pass that records every tier's output for every frame.

The five strategies and every gate setting only differ in which tiers a frame
visits. So instead of running the hardware once per strategy and per parameter
value, this run sends EVERY frame through every tier once and records:

  * p_i, p_ES, p_CS  (the three softmax vectors)            -> logs/trace.npz
  * the measured times: ViT-Small, E2LM to Edge and Cloud,  -> logs/trace.csv
    ViT-Base, ViT-Large, and the IoT stopwatch for each MQTT path
    (Edge round trip, Edge -> Cloud cascade, direct Cloud round trip)
  * the simulated Rayleigh and backhaul delays for the frame

make_plot_data.py (laptop) then replays HCCLIO, CPO, Edge only, Cloud only and
Local only from this trace for any tau_conf, tau_lat or mu_ES, exactly as the
live code decides, and writes the CSV for each plot.

Start the Edge and Cloud tiers first, then on the Pi:

    python outputs/iot_device_tier/trace.py --dataset ~/HCCLIO
    python outputs/iot_device_tier/trace.py --dataset ~/HCCLIO --label vit_tiny   # another model set
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
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

from common.messages import topics  # noqa: E402

log = logging.getLogger("trace")

COLUMNS = [
    "frame_id", "image_file", "ground_truth_idx", "ok",
    "local_inference_ms", "E2LM_edge_ms", "wireless_delay_ms", "E2LM_cloud_ms",
    "edge_inference_ms", "E2LM_cloud_edge_ms", "backhaul_delay_ms", "cloud_inference_ms",
    "cascade_wallclock_ms", "edge_ping_ms", "cloud_ping_ms",
    "model_iot", "model_edge", "model_cloud", "A_i", "A_ES", "A_CS", "T_i_ms",
]


def short(name: str) -> str:
    """vit_small_patch16_224.augreg_in21k_ft_in1k -> vit_small"""
    return re.sub(r"_patch.*", "", name)


def trace_frame(f, vit, e2lm, mqtt, t, cfg) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray]:
    local = vit.infer(f.jpeg)
    e2lm_edge, channel = e2lm.edge_probe()
    e2lm_cloud = e2lm.cloud_ms()
    base = {"frame_id": f.frame_id, "jpeg": f.jpeg, "p_i": local.p_i, "path": "IoT"}

    # Edge -> Cloud cascade (the Edge is told to always cascade); also returns p_ES, p_CS
    t0 = time.perf_counter()
    ans = mqtt.request(t["edge_request"], {**base, "strategy": "trace", "route": "trace", "timings": {}},
                       cfg.response_timeout_s)
    cascade_wall = (time.perf_counter() - t0) * 1000.0
    # MQTT round trips with the same payload but no inference
    pings = {}
    for key, topic in (("edge_ping_ms", t["edge_request"]), ("cloud_ping_ms", t["cloud_request"])):
        t0 = time.perf_counter()
        r = mqtt.request(topic, {**base, "strategy": "ping", "route": "ping", "timings": {}}, cfg.response_timeout_s)
        pings[key] = (time.perf_counter() - t0) * 1000.0 if r is not None else float("nan")

    tm = (ans or {}).get("timings", {})
    row = {
        "frame_id": f.frame_id, "image_file": f.image_file, "ground_truth_idx": f.ground_truth_idx,
        "ok": int(ans is not None and ans.get("p_cs") is not None),
        "local_inference_ms": local.t_ms, "E2LM_edge_ms": e2lm_edge,
        "wireless_delay_ms": float(channel.get("wireless_delay_ms", 0.0)), "E2LM_cloud_ms": e2lm_cloud,
        "edge_inference_ms": tm.get("edge_inference_ms", float("nan")),
        "E2LM_cloud_edge_ms": tm.get("E2LM_cloud_ms", float("nan")),
        "backhaul_delay_ms": tm.get("backhaul_delay_ms", float("nan")),
        "cloud_inference_ms": tm.get("cloud_inference_ms", float("nan")),
        "cascade_wallclock_ms": cascade_wall, **pings,
    }
    return row, (ans or {}).get("p_es"), (ans or {}).get("p_cs"), local.p_i


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["timm", "stub"], default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None,
                    help="name of this model set (default: the three model names); output logs/trace[_label].csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load(args.config, args.backend)
    mods = cfg.raw["models"]
    names = {k: short(mods[k]["name"]) for k in ("iot", "edge", "cloud")}
    acc = {"A_i": float(mods["iot"]["accuracy"]), "A_ES": float(mods["edge"]["accuracy"]),
           "A_CS": float(mods["cloud"]["accuracy"])}
    frames = load_dataset(args.dataset or cfg.dataset_root, args.limit or cfg.n_frames)
    suffix = f"_{args.label}" if args.label else ""
    out_csv = cfg.csv_path.with_name(f"trace{suffix}.csv")

    vit = ViTSmall(cfg.model_name, cfg.backend)
    e2lm = E2LM(cfg)
    t = topics(cfg.topic_prefix)
    mqtt = MqttClient(cfg.broker_host, cfg.broker_port, "iot-pi4-trace",
                      t["iot_response"].format(client="pi4-trace"), cfg.qos)
    p_i_all, p_es_all, p_cs_all = [], [], []
    t_start = time.time()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in frames:
            row, p_es, p_cs, p_i = trace_frame(f, vit, e2lm, mqtt, t, cfg)
            row.update({"model_iot": names["iot"], "model_edge": names["edge"], "model_cloud": names["cloud"],
                        **acc, "T_i_ms": cfg.t_i_ms})
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            nan = np.full(len(p_i), np.nan, dtype=np.float32)
            p_i_all.append(np.asarray(p_i, np.float32))
            p_es_all.append(np.asarray(p_es, np.float32) if p_es is not None else nan)
            p_cs_all.append(np.asarray(p_cs, np.float32) if p_cs is not None else nan)
            log.info("frame %4d  ok=%d  cascade %.0f ms  edge ping %.0f ms  cloud ping %.0f ms", f.frame_id,
                     row["ok"], row["cascade_wallclock_ms"], row["edge_ping_ms"], row["cloud_ping_ms"])
    np.savez_compressed(out_csv.with_suffix(".npz"), p_i=np.stack(p_i_all), p_es=np.stack(p_es_all),
                        p_cs=np.stack(p_cs_all))
    mqtt.close()
    log.info("done in %.1f s -> %s (+ .npz)", time.time() - t_start, out_csv)


if __name__ == "__main__":
    main()
