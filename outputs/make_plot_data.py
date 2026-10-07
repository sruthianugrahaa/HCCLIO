"""make_plot_data.py - CSV files for the five QoE plots, for every strategy.

Reads the trace written by iot_device_tier/trace.py (logs/trace.csv + .npz:
every tier's softmax and every measured / simulated delay for every frame)
and replays the five strategies exactly as the live code decides:

  HCCLIO      proposed: confidence gate at the IoT device, latency gate, Edge
              ensemble, Edge -> Cloud cascade or direct Cloud
  CPO         confidence-based offloading (the DCI benchmark): confidence gate
              only, every offloaded frame goes to the Edge, no latency gate
  Edge only   every frame -> Edge, ViT-Base alone
  Cloud only  every frame -> Cloud, ViT-Large alone
  Local only  ViT-Small on the IoT device, nothing offloaded

QoE is from the IoT device's side, as in the live runs:
  T_E2E = IoT-side measured times (inference, E2LM probes, MQTT transfer from the
          IoT stopwatch) + simulated Rayleigh and backhaul delays
  Q     = (1 - T_E2E / T_i^comp) * A_x,  Q = 0 (Discard) if T_E2E >= T_i^comp

Output (outputs/reports/plot_data/), one row per strategy and x value:
  qoe_vs_mu_edge.csv            plot 1: Avg QoE vs Edge processing rate mu_ES, 0.01-20 tasks/s
                                (the Edge is an M/M/1 queue, see edge_queue below)
  qoe_vs_conf_threshold.csv     plot 2: Avg QoE vs confidence threshold tau_conf
  qoe_vs_latency_threshold.csv  plot 3: Avg QoE vs latency threshold tau_lat
  qoe_vs_models.csv             plot 4: Avg QoE per model set (one trace file per set)
  qoe_benchmark_comparison.csv  plot 5: QoE, accuracy, latency, discards per strategy
  per_frame/per_frame_<strategy>.csv
                                one row per image: tier, path, prediction, correct,
                                confidences, every delay on the path taken (incl. the
                                MQTT transfer the IoT stopwatch saw), T_E2E, A_x, Q_x

    python outputs/make_plot_data.py                      # every logs/trace*.csv
    python outputs/make_plot_data.py --trials 500
    python outputs/make_plot_data.py --edge-model exp     # Edge time ~ Exp(1/mu), no queue, gate unaware

Edge model for plot 1 (--edge-model mm1, the default)
  Frames reach the Edge at lambda_ES = arrival_rate * (share of frames the strategy
  sends to the Edge); arrival_rate is mu_sweep.arrival_rate in hcclio.yaml.
  * Time a frame spends at the Edge (queue + ViT-Base) ~ Exp(mean 1/(mu - lambda_ES));
    if mu <= lambda_ES the queue grows without bound and those frames are discarded.
  * The E2LM probe to the Edge waits in the same queue, so HCCLIO's latency gate sees
    delta_wl + delta_E2LM + W_q, with W_q = lambda_ES / (mu (mu - lambda_ES)).
    HCCLIO's own lambda_ES depends on how many frames pass that gate, so it is solved
    as a fixed point. CPO and Edge only have no latency gate and keep sending to the Edge.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

STRATEGIES = ["HCCLIO", "CPO", "Edge only", "Cloud only", "Local only"]
TAU_CONF = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
TAU_LAT_MS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
TIERS = ["IoT", "Edge", "Cloud", "Fallback-IoT", "Discard"]
MU_ES = [float(f"{x:.4g}") for x in np.geomspace(0.01, 20, 25)]   # tasks/s
UNSTABLE_MS = 1e7   # time at an Edge whose queue never empties (always a Discard)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_trace(csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    npz = np.load(csv_path.with_suffix(".npz"))
    ok = np.array([r["ok"] == "1" for r in rows])
    num = [c for c in rows[0] if c.endswith("_ms") or c in ("ground_truth_idx", "A_i", "A_ES", "A_CS")]
    t = {c: np.array([_f(r[c]) for r in rows])[ok] for c in num}
    t.update(p_i=npz["p_i"][ok].astype(np.float64), p_es=npz["p_es"][ok].astype(np.float64),
             p_cs=npz["p_cs"][ok].astype(np.float64), n_failed=int((~ok).sum()))
    t["models"] = {k: rows[0][f"model_{k}"] for k in ("iot", "edge", "cloud")}
    t["frame_id"] = np.array([int(r["frame_id"]) for r in rows])[ok]
    t["image_file"] = np.array([r["image_file"] for r in rows])[ok]
    # transfer overheads seen by the IoT stopwatch (MQTT up + down, no inference)
    t["ovh_edge"] = np.nan_to_num(t["edge_ping_ms"])
    t["ovh_direct"] = np.nan_to_num(t["cloud_ping_ms"])
    t["ovh_cascade"] = np.maximum(0.0, t["cascade_wallclock_ms"] - t["edge_inference_ms"]
                                  - t["cloud_inference_ms"] - t["E2LM_cloud_edge_ms"])
    return t


def _top(p):
    y = p.argmax(axis=1)
    return p[np.arange(len(p)), y], y


def ensembles(t: dict, cfg: dict) -> dict:
    """Every confidence / prediction the gates can see, once per frame."""
    w, use_ens = cfg["weights"], bool(cfg["dci"]["use_ensemble"])
    we, wc, wd = w["edge"], w["cloud"], w["cloud_direct"]
    e = {}
    e["c_i"], e["y_i"] = _top(t["p_i"])
    e["c_e"], e["y_e"] = _top(we["w_i"] * t["p_i"] + we["w_ES"] * t["p_es"])
    e["c_c"], e["y_c"] = _top(wc["w_i"] * t["p_i"] + wc["w_ES"] * t["p_es"] + wc["w_CS"] * t["p_cs"])
    e["c_d"], e["y_d"] = _top(wd["w_i"] * t["p_i"] + wd["w_CS"] * t["p_cs"])
    e["c_es"], e["y_es"] = _top(t["p_es"])
    e["c_cs"], e["y_cs"] = _top(t["p_cs"])
    e["cpo_ensemble"] = use_ens
    return e


def replay(t: dict, e: dict, strategy: str, tau_conf: float, tau_lat: float, t_i: float,
           edge_ms: np.ndarray | None = None, edge_wait_ms: float = 0.0) -> dict:
    """Per-frame tier, prediction and T_E2E for one strategy and one parameter setting.

    edge_ms replaces the measured Edge time; edge_wait_ms is the Edge queueing delay the
    E2LM probe sees, added to HCCLIO's latency-gate signal."""
    n = len(t["local_inference_ms"])
    edge_t = t["edge_inference_ms"] if edge_ms is None else edge_ms
    loc, e2e_e, wl = t["local_inference_ms"], t["E2LM_edge_ms"], t["wireless_delay_ms"]
    bh, cloud_t, e2c, e2ce = t["backhaul_delay_ms"], t["cloud_inference_ms"], t["E2LM_cloud_ms"], t["E2LM_cloud_edge_ms"]
    T_edge = loc + e2e_e + wl + edge_t + t["ovh_edge"]
    T_casc = loc + e2e_e + wl + edge_t + e2ce + bh + cloud_t + t["ovh_cascade"]
    T_dir = loc + e2e_e + wl + e2c + bh + cloud_t + t["ovh_direct"]

    tier = np.empty(n, dtype=object)
    path = np.empty(n, dtype=object)
    pred = np.zeros(n, dtype=int)
    T = np.zeros(n)
    if strategy == "Local only":
        tier[:], path[:], pred[:], T[:] = "IoT", "IoT", e["y_i"], loc
    elif strategy == "Edge only":
        tier[:], path[:], pred[:], T[:] = "Edge", "Edge only", e["y_es"], e2e_e + wl + edge_t + t["ovh_edge"]
    elif strategy == "Cloud only":
        tier[:], path[:], pred[:], T[:] = "Cloud", "Cloud only", e["y_cs"], e2c + wl + bh + cloud_t + t["ovh_direct"]
    else:
        ce, ye, cc, yc = e["c_e"], e["y_e"], e["c_c"], e["y_c"]
        if strategy == "CPO" and not e["cpo_ensemble"]:
            ce, ye, cc, yc = e["c_es"], e["y_es"], e["c_cs"], e["y_cs"]
        local_ok = e["c_i"] >= tau_conf
        to_edge = np.ones(n, bool) if strategy == "CPO" else (wl + e2e_e + edge_wait_ms <= tau_lat)
        edge_ok, casc_ok, dir_ok = ce >= tau_conf, cc >= tau_conf, e["c_d"] >= tau_conf
        cases = [
            (local_ok, "IoT", "IoT", e["y_i"], loc),
            (~local_ok & to_edge & edge_ok, "Edge", "IoT->Edge", ye, T_edge),
            (~local_ok & to_edge & ~edge_ok & casc_ok, "Cloud", "IoT->Edge->Cloud", yc, T_casc),
            (~local_ok & to_edge & ~edge_ok & ~casc_ok, "Fallback-IoT", "IoT->Edge->Cloud", e["y_i"], T_casc),
            (~local_ok & ~to_edge & dir_ok, "Cloud", "IoT->Cloud", e["y_d"], T_dir),
            (~local_ok & ~to_edge & ~dir_ok, "Fallback-IoT", "IoT->Cloud", e["y_i"], T_dir),
        ]
        for m, name, pth, y, tt in cases:
            tier[m], path[m], pred[m], T[m] = name, pth, y[m], tt[m]

    a = np.where(tier == "Edge", t["A_ES"], np.where(tier == "Cloud", t["A_CS"], t["A_i"]))
    discard = T >= t_i
    q = np.where(discard, 0.0, (1.0 - T / t_i) * a)
    tier = np.where(discard, "Discard", tier)
    return {"Q": q, "T": T, "tier": tier, "path": path, "pred": pred, "A": a,
            "correct": pred == t["ground_truth_idx"].astype(int), "discard": discard}


# which delay components each path's T_E2E is made of (others are left empty in the per-frame CSV)
PATH_PARTS = {
    "IoT": ["local_inference_ms"],
    "IoT->Edge": ["local_inference_ms", "E2LM_edge_ms", "wireless_delay_ms", "edge_inference_ms", "ovh_edge"],
    "IoT->Edge->Cloud": ["local_inference_ms", "E2LM_edge_ms", "wireless_delay_ms", "edge_inference_ms",
                         "E2LM_cloud_edge_ms", "backhaul_delay_ms", "cloud_inference_ms", "ovh_cascade"],
    "IoT->Cloud": ["local_inference_ms", "E2LM_edge_ms", "wireless_delay_ms", "E2LM_cloud_ms",
                   "backhaul_delay_ms", "cloud_inference_ms", "ovh_direct"],
    "Edge only": ["E2LM_edge_ms", "wireless_delay_ms", "edge_inference_ms", "ovh_edge"],
    "Cloud only": ["E2LM_cloud_ms", "wireless_delay_ms", "backhaul_delay_ms", "cloud_inference_ms", "ovh_direct"],
}
PART_COLUMNS = ["local_inference_ms", "E2LM_edge_ms", "wireless_delay_ms", "edge_inference_ms",
                "E2LM_cloud_edge_ms", "E2LM_cloud_ms", "backhaul_delay_ms", "cloud_inference_ms"]
OVH = {"ovh_edge", "ovh_cascade", "ovh_direct"}


def edge_wait_ms(mu: float, lam: float) -> float:
    """M/M/1 mean waiting time in the queue (ms) at service rate mu, arrival rate lam (per s)."""
    return 1000.0 * lam / (mu * (mu - lam)) if lam < mu else float("inf")


def edge_queue(t: dict, e: dict, strategy: str, mu: float, arrival_rate: float, tau_conf: float,
               tau_lat: float) -> tuple[float, float]:
    """(lambda_ES, W_q ms) at the Edge for one strategy at Edge rate mu (M/M/1)."""
    if strategy in ("Local only", "Cloud only"):
        return 0.0, 0.0
    if strategy == "Edge only":
        lam = arrival_rate
        return lam, edge_wait_ms(mu, lam)
    offload = e["c_i"] < tau_conf
    if strategy == "CPO":
        lam = arrival_rate * float(offload.mean())
        return lam, edge_wait_ms(mu, lam)
    gate = t["wireless_delay_ms"] + t["E2LM_edge_ms"]

    def share(lam):  # share of frames HCCLIO sends to the Edge when it carries lam
        return float((offload & (gate + edge_wait_ms(mu, lam) <= tau_lat)).mean())

    lo, hi = 0.0, share(0.0)  # share(f * arrival_rate) - f falls as f grows: bisect
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if share(mid * arrival_rate) > mid else (lo, mid)
    lam = lo * arrival_rate
    return lam, edge_wait_ms(mu, lam)


def per_frame_rows(t: dict, e: dict, strategy: str, r: dict, tau_conf: float, tau_lat: float, t_i: float) -> list[dict]:
    """One row per image: decision, every delay on the path taken, T_E2E and Q."""
    rows = []
    for k in range(len(r["Q"])):
        parts = PATH_PARTS[r["path"][k]]
        row = {"frame_id": int(t["frame_id"][k]), "image_file": t["image_file"][k],
               "ground_truth_idx": int(t["ground_truth_idx"][k]), "strategy": strategy,
               "tier": r["tier"][k], "path": r["path"][k], "prediction_idx": int(r["pred"][k]),
               "correct": int(r["correct"][k]),
               "c_i": float(e["c_i"][k]), "c_edge_ensemble": float(e["c_e"][k]),
               "c_cloud_ensemble": float(e["c_c"][k]), "c_cloud_direct": float(e["c_d"][k])}
        for c in PART_COLUMNS:
            row[c] = float(t[c][k]) if c in parts else ""
        ovh = [p for p in parts if p in OVH]
        row["mqtt_transfer_ms"] = float(t[ovh[0]][k]) if ovh else ""
        row.update({"e2e_latency_ms": float(r["T"][k]), "A_x": float(r["A"][k]), "T_i_ms": t_i,
                    "Q_x": float(r["Q"][k]), "tau_conf": tau_conf, "tau_lat_ms": tau_lat})
        rows.append(row)
    return rows


def metrics(r: dict) -> dict:
    n = len(r["Q"])
    out = {
        "mean_QoE": float(r["Q"].mean()),
        "ci95_QoE": float(1.96 * r["Q"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "accuracy": float(r["correct"].mean()),
        "mean_e2e_ms": float(r["T"].mean()),
        "p95_e2e_ms": float(np.percentile(r["T"], 95)),
        "discard_rate": float(r["discard"].mean()),
    }
    for k in TIERS:
        out[f"share_{k}"] = float((r["tier"] == k).mean())
    return out


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def make_all(traces: dict[str, Path], cfg: dict, out_dir: Path, trials: int = 200, seed: int = 2026,
             mu_values=None, edge_model: str = "mm1") -> list[Path]:
    g, q = cfg["gates"], cfg["qoe"]
    tau_c, tau_l = float(g["tau_conf"]), float(g["tau_lat_ms"])
    mu_values = mu_values or MU_ES
    arrival = float(cfg["mu_sweep"].get("arrival_rate", 1.0))
    main_label = next(iter(traces))
    loaded = {lab: load_trace(p) for lab, p in traces.items()}
    t = loaded[main_label]
    e = ensembles(t, cfg)
    t_i = float(np.nanmedian(t["T_i_ms"])) if np.isfinite(t["T_i_ms"]).any() else float(q["t_i_ms"])
    n = len(t["local_inference_ms"])
    files = []

    # plot 5: comparison at the configured gates
    rows = [{"strategy": s, "frames": n, "tau_conf": tau_c, "tau_lat_ms": tau_l, "T_i_ms": t_i,
             **metrics(replay(t, e, s, tau_c, tau_l, t_i))} for s in STRATEGIES]
    files.append(write(out_dir / "qoe_benchmark_comparison.csv", rows))

    # per image, per strategy (1000 rows each) at the configured gates
    for s in STRATEGIES:
        r = replay(t, e, s, tau_c, tau_l, t_i)
        name = s.lower().replace(" ", "_")
        files.append(write(out_dir / "per_frame" / f"per_frame_{name}.csv",
                           per_frame_rows(t, e, s, r, tau_c, tau_l, t_i)))

    # plot 1: QoE vs mu_ES (M/M/1 Edge by default, see the module docstring)
    rng = np.random.default_rng(seed)
    rows = []
    for mu in mu_values:
        mu = float(mu)
        unit = [rng.exponential(1.0, n) for _ in range(trials)]
        for s in STRATEGIES:
            if edge_model == "mm1":
                lam, wq = edge_queue(t, e, s, mu, arrival, tau_c, tau_l)
                mean_ms = 1000.0 / (mu - lam) if lam < mu else UNSTABLE_MS
                samples = [np.minimum(u * mean_ms, UNSTABLE_MS) if lam < mu else np.full(n, UNSTABLE_MS)
                           for u in unit]
                wq = min(wq, UNSTABLE_MS)
            else:
                lam, wq, mean_ms = 0.0, 0.0, 1000.0 / mu
                samples = [u * mean_ms for u in unit]
            per = [replay(t, e, s, tau_c, tau_l, t_i, edge_ms=x, edge_wait_ms=wq) for x in samples]
            qs = np.array([r["Q"].mean() for r in per])
            rows.append({"strategy": s, "mu_ES": mu, "edge_model": edge_model, "lambda_ES": lam,
                         "edge_wait_ms": wq, "mean_edge_time_ms": mean_ms if s not in ("Local only", "Cloud only") else 0.0,
                         "mean_QoE": float(qs.mean()),
                         "ci95_QoE": float(1.96 * qs.std(ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0,
                         "discard_rate": float(np.mean([r["discard"].mean() for r in per])),
                         "share_Edge_path": float(np.mean([np.isin(r["path"], ["IoT->Edge", "IoT->Edge->Cloud",
                                                                                "Edge only"]).mean() for r in per])),
                         "mean_e2e_ms": float(np.mean([r["T"].mean() for r in per])), "trials": trials})
    files.append(write(out_dir / "qoe_vs_mu_edge.csv", rows))

    # plot 2: QoE vs confidence threshold
    rows = [{"strategy": s, "tau_conf": tc, **metrics(replay(t, e, s, tc, tau_l, t_i))}
            for tc in TAU_CONF for s in STRATEGIES]
    files.append(write(out_dir / "qoe_vs_conf_threshold.csv", rows))

    # plot 3: QoE vs latency threshold
    rows = [{"strategy": s, "tau_lat_ms": tl, **metrics(replay(t, e, s, tau_c, float(tl), t_i))}
            for tl in TAU_LAT_MS for s in STRATEGIES]
    files.append(write(out_dir / "qoe_vs_latency_threshold.csv", rows))

    # plot 4: one model set per trace file
    rows = []
    for lab, tt in loaded.items():
        ee = ensembles(tt, cfg)
        m = tt["models"]
        for s in STRATEGIES:
            rows.append({"model_set": lab, "model_iot": m["iot"], "model_edge": m["edge"], "model_cloud": m["cloud"],
                         "strategy": s, **metrics(replay(tt, ee, s, tau_c, tau_l, t_i))})
    files.append(write(out_dir / "qoe_vs_models.csv", rows))
    if t["n_failed"]:
        print(f"note: {t['n_failed']} frames of {main_label} had no Cloud/Edge reply and were left out")
    return files


def find_traces(log_dir: Path) -> dict[str, Path]:
    found = sorted(log_dir.glob("trace*.csv"))
    out = {}
    for p in found:
        if not p.with_suffix(".npz").exists():
            continue
        label = p.stem[len("trace_"):] if p.stem.startswith("trace_") else None
        if label is None:  # trace.csv: name it after its models
            with open(p, newline="", encoding="utf-8") as fh:
                r = next(csv.DictReader(fh), None)
            label = f"{r['model_iot']}+{r['model_edge']}+{r['model_cloud']}" if r else "trace"
        out[label] = p
    if any(p.stem == "trace" for p in out.values()):  # the default trace first (used by plots 1, 2, 3, 5)
        out = dict(sorted(out.items(), key=lambda kv: kv[1].stem != "trace"))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # outputs/ (holds common/)
    from common.settings import load_settings, resolve_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=200, help="Monte-Carlo trials per mu_ES value")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--edge-model", choices=["mm1", "exp"], default="mm1",
                    help="plot 1: mm1 = Edge queue, HCCLIO gate sees its wait (default); exp = no queue")
    a = ap.parse_args()
    cfg = load_settings(a.config)
    log_dir = resolve_path(cfg["logging"]["log_dir"])
    traces = find_traces(log_dir)
    if not traces:
        sys.exit(f"no trace*.csv + .npz in {log_dir}: run iot_device_tier/trace.py on the Pi and copy logs/ here")
    out = Path(a.out_dir) if a.out_dir else resolve_path(cfg["logging"]["report_dir"]) / "plot_data"
    print("traces:", ", ".join(traces))
    for f in make_all(traces, cfg, out, a.trials, int(cfg["mu_sweep"]["seed"]), edge_model=a.edge_model):
        print("wrote", f)
