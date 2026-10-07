"""Tests for the per-tier code under outputs/. Run: python -m pytest"""

import csv
import importlib.util
import math
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(OUT))      # common/
sys.path.insert(0, str(ROOT))     # scripts/

from common.messages import pack, unpack  # noqa: E402
from common.settings import load_settings  # noqa: E402
from common.stub_vit import StubClassifier, jpeg_ground_truth  # noqa: E402


def load(rel: str):
    """Import one tier file by path under a unique module name (tiers reuse file names)."""
    path = OUT / rel
    name = rel.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def cfg():
    return load_settings()


# ------------------------------------------------------------------ common
def test_weights(cfg):
    w = cfg["weights"]
    assert w["edge"] == {"w_i": 0.44, "w_ES": 0.56}
    assert math.isclose(w["cloud_direct"]["w_i"], 0.28 / 0.66)
    with pytest.raises(ValueError):
        load_settings(overrides={"weights": {"edge": {"w_i": 0.5, "w_ES": 0.6}}})


def test_yaml_numbers_are_numbers(cfg):
    for section in ("wireless", "backhaul"):
        for k, v in cfg[section].items():
            assert not (isinstance(v, str) and v[:1].isdigit()), f"{section}.{k} parsed as text: {v!r}"


def test_msgpack_roundtrip():
    a = np.random.default_rng(0).random(1000).astype(np.float32)
    out = unpack(pack({"p": a, "jpeg": b"\xff\xd8x", "t": {"x": 1.5}}))
    assert np.array_equal(out["p"], a) and out["jpeg"] == b"\xff\xd8x" and out["t"]["x"] == 1.5


def test_stub_deterministic(tmp_path):
    dl = load("iot_device_tier/download_dataset.py")
    dl.make_synthetic_dataset(tmp_path, 3)
    frames = load("iot_device_tier/load_dataset.py").load_dataset(str(tmp_path))
    f = frames[0]
    assert jpeg_ground_truth(f.jpeg) == f.ground_truth_idx
    p1, _ = StubClassifier("iot", 0.75).predict(f.jpeg)
    p2, _ = StubClassifier("iot", 0.75).predict(f.jpeg)
    assert np.array_equal(p1, p2) and math.isclose(float(p1.sum()), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------- channel models
def test_rayleigh(cfg):
    er = load("edge_server_tier/edge_rayleigh_delay.py")
    assert math.isclose(er.pathloss_db(3.5e9, 10, True), 31.84 + 19 * math.log10(3.5) + 20)
    assert math.isclose(er.pathloss_db(3.5e9, 10, False), 32.4 + 20 * math.log10(3.5) + 30)
    assert math.isclose(er.p_los(10, 0.5, 18), 0.5 ** (10 / 18))
    ch = er.EdgeRayleighDelay(cfg["wireless"], seed=1)
    s = ch.sample()
    assert math.isclose(s["tx_delay_ms"], 1e6 / s["rate_bps"] * 1000)
    assert math.isclose(s["rate_bps"], 20e6 * math.log2(1 + 10 ** (s["snr_db"] / 10)), rel_tol=1e-9)
    d = np.array([ch.delay_ms() for _ in range(5000)])
    assert (d > 0).all() and np.median(d) < 20


def test_backhaul(cfg):
    bh = load("cloud_server_tier/backhaul_delay.py").BackhaulDelay(cfg["backhaul"], seed=2)
    assert bh.m_bh == math.floor((1 + 1.28 * 4) * 1.5 + 2 * 1.0) == 11
    assert math.isclose(bh.theta_ms, 0.2 + 1e6 * 1.5e-6)
    d = np.array([bh.delay_ms() for _ in range(20000)])
    assert abs(d.mean() - bh.mean_ms()) / bh.mean_ms() < 0.02


# --------------------------------------------------------------- QoE / mu
def test_qoe_and_mu_model():
    q = load("iot_device_tier/qoe.py")
    assert q.qoe(250, 1000, 0.8) == (pytest.approx(0.6), False)
    assert q.qoe(1000, 1000, 0.8) == (0.0, True)
    assert q.e2e_latency_ms({"local_inference_ms": 10, "wireless_delay_ms": 2.5, "cloud_inference_ms": None}) == 12.5
    assert q.iot_e2e_ms(100.0, {"local_inference_ms": 40, "wireless_delay_ms": 2.5, "backhaul_delay_ms": 7.5}) == 110.0
    mu = q.service_time_s(q.complexity_from_model("i", 1e6), 1e6, q.rate_from_measurement("i", 200.0))
    assert mu == pytest.approx(0.2)
    assert q.mean_system_time_s(0.2, 1.0, 2.0) == pytest.approx(1 / 3)
    assert math.isinf(q.mean_system_time_s(0.2, 1.0, 5.0))


# ------------------------------------------------------ ensembles and gates
class FakeModel:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float32)

    def predict(self, jpeg):
        return self.p, 1.0


class C:
    tau_conf = 0.8
    w_i, w_ES = 0.44, 0.56
    w_cascade = {"w_i": 0.28, "w_ES": 0.34, "w_CS": 0.38}
    w_direct = {"w_i": 0.28 / 0.66, "w_CS": 0.38 / 0.66}
    dci_use_ensemble = True


def test_edge_ensemble_gate():
    ei = load("edge_server_tier/inference.py")
    r = ei.edge_inference(FakeModel([0.1, 0.9]), b"", np.array([0.7, 0.3]), "hcclio", C)
    assert r.y_edge == 1 and math.isclose(r.c_edge, 0.44 * 0.3 + 0.56 * 0.9, rel_tol=1e-6) and not r.answer_here
    r = ei.edge_inference(FakeModel([0.05, 0.95]), b"", np.array([0.2, 0.8]), "hcclio", C)
    assert r.answer_here
    assert ei.edge_inference(FakeModel([0.4, 0.6]), b"", None, "edge_only", C).answer_here


def test_cloud_ensemble_and_fallback():
    ci = load("cloud_server_tier/inference.py")
    p_i, p_es, p_cs = np.array([0.6, 0.4]), np.array([0.3, 0.7]), np.array([0.1, 0.9])
    r = ci.cloud_inference(FakeModel(p_cs), {"jpeg": b"", "p_i": p_i, "p_es": p_es, "strategy": "hcclio"}, C)
    assert math.isclose(r.c_cloud, 0.28 * 0.4 + 0.34 * 0.7 + 0.38 * 0.9, rel_tol=1e-6)
    assert r.tier == "Fallback-IoT" and r.prediction == 0  # below 0.8 -> y_i
    r = ci.cloud_inference(FakeModel([0.0, 1.0]), {"jpeg": b"", "p_i": np.array([0.1, 0.9]), "strategy": "hcclio"}, C)
    assert r.tier == "Cloud" and math.isclose(r.c_cloud, 0.28 / 0.66 * 0.9 + 0.38 / 0.66)
    r = ci.cloud_inference(FakeModel([0.4, 0.6]), {"jpeg": b"", "p_i": None, "strategy": "cloud_only"}, C)
    assert r.tier == "Cloud"


# -------------------------------------------------------------------- E2LM
def test_e2lm_probe_and_channel_handoff():
    edge = load("edge_server_tier/e2lm.py")
    cloud = load("cloud_server_tier/e2lm_server.py")
    iot = load("iot_device_tier/e2lm.py")
    ep, cp = free_port(), free_port()
    es = edge.E2LMServer(ep, 1000, lambda: {"wireless_delay_ms": 3.25}).start_background()
    cs = cloud.E2LMServer(cp, 1000).start_background()
    try:
        ms, chan = iot.e2lm_probe("127.0.0.1", ep)
        assert 0 < ms < 500 and chan == {"wireless_delay_ms": 3.25}
        ms, chan = iot.e2lm_probe("127.0.0.1", cp)
        assert 0 < ms < 500 and chan == {}
        assert 0 < edge.probe_ms("127.0.0.1", cp) < 500
    finally:
        for s in (es, cs):
            s.shutdown()
            s.server_close()
    assert iot.e2lm_delay_ms("127.0.0.1", free_port(), timeout_s=0.5) == 500.0


def test_timm_vit_code_path(monkeypatch):
    timm = pytest.importorskip("timm")
    pytest.importorskip("torch")
    create = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained=False, **kw: create(name, pretrained=False, **kw))
    inf = load("iot_device_tier/inference.py")
    r = inf.ViTSmall()  # random weights, real preprocessing + forward pass
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (200, 30, 30)).save(buf, format="JPEG")
    res = r.infer(buf.getvalue())
    assert res.p_i.shape == (1000,) and math.isclose(float(res.p_i.sum()), 1.0, rel_tol=1e-4) and res.t_ms > 0


def test_class_indices_match_timm():
    pytest.importorskip("timm")
    from timm.data import ImageNetInfo

    from common.classes import COBOT_CLASSES

    info = ImageNetInfo()
    for name, idx in COBOT_CLASSES.items():
        assert name.split("_")[0] in info.index_to_description(idx).lower().replace(" ", "_")


# ------------------------------------------------------- whole system (MQTT)
needs_broker = pytest.mark.skipif(shutil.which("mosquitto") is None, reason="needs mosquitto")


@pytest.fixture(scope="module")
def broker():
    port = free_port()
    p = subprocess.Popen(["mosquitto", "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time

    time.sleep(0.5)
    yield port
    p.terminate()


def rows_of(path):
    return list(csv.DictReader(open(path, newline="")))


def f(x):
    return float(x) if x not in ("", None) else 0.0


@needs_broker
def test_all_strategies_end_to_end(broker, tmp_path_factory):
    from scripts.loopback import run

    out = run(tmp_path_factory.mktemp("dry"), frames=60, broker_port=broker)
    logs = out / "logs"
    delay_cols = ["local_inference_ms", "wireless_delay_ms", "E2LM_edge_ms", "edge_inference_ms",
                  "backhaul_delay_ms", "E2LM_cloud_ms", "cloud_inference_ms"]
    for name in ("qoe_coclio", "qoe_local_only", "qoe_edge_only", "qoe_cloud_only", "qoe_distributed_benchmark"):
        rows = rows_of(logs / f"{name}.csv")
        assert len(rows) == 60, name
        assert not {r["tier"] for r in rows} & {"Timeout"}, name
        for r in rows:
            assert f(r["e2e_sum_ms"]) == pytest.approx(sum(f(r[c]) for c in delay_cols), abs=1e-3)
            # QoE uses the IoT device's own stopwatch + the simulated link delays
            sim = f(r["wireless_delay_ms"]) + f(r["backhaul_delay_ms"])
            assert f(r["e2e_latency_ms"]) == pytest.approx(f(r["iot_wallclock_ms"]) + sim, abs=1e-3)
            assert f(r["e2e_latency_ms"]) >= f(r["e2e_sum_ms"]) - 1.0   # real waiting >= its measured parts
            assert (f(r["backhaul_delay_ms"]) > 0) == ("Cloud" in r["path"])   # drawn by the Cloud
            if r["path"] != "IoT":
                assert f(r["wireless_delay_ms"]) > 0                            # sent by the Edge
    hc = rows_of(logs / "qoe_coclio.csv")
    assert all(f(r["iot_confidence"]) >= 0.8 for r in hc if r["tier"] == "IoT")
    assert all(f(r["aggregated_conf"]) >= 0.8 for r in hc if r["tier"] in ("Edge", "Cloud"))
    assert {r["tier"] for r in rows_of(logs / "qoe_local_only.csv")} == {"IoT"}
    eo = rows_of(logs / "qoe_edge_only.csv")
    assert {r["tier"] for r in eo} == {"Edge"} and all(r["local_inference_ms"] == "" for r in eo)
    assert all(r["path"] == "IoT->Cloud" for r in rows_of(logs / "qoe_cloud_only.csv"))
    for tier in ("i", "ES", "CS"):
        assert (out / "reports" / f"mu_sweep_{tier}.csv").exists()
    assert (out / "reports" / "summary.md").exists()


@needs_broker
def test_latency_gate_fail_goes_direct_to_cloud(broker, tmp_path_factory):
    from scripts.loopback import run

    out = run(tmp_path_factory.mktemp("latfail"), frames=40, broker_port=broker,
              overrides={"gates": {"tau_lat_ms": 0.0}}, strategies=["hcclio", "dci"], analysis=False)
    off = [r for r in rows_of(out / "logs" / "qoe_coclio.csv") if r["tier"] != "IoT"]
    assert off and all(r["path"] == "IoT->Cloud" and "lat_fail" in r["route"] for r in off)
    dci = [r for r in rows_of(out / "logs" / "qoe_distributed_benchmark.csv") if r["tier"] != "IoT"]
    assert dci and all(r["path"].startswith("IoT->Edge") for r in dci)  # DCI has no latency gate


def test_mu_sweep_monotone(tmp_path):
    ms = load("run_mu_sweep.py")
    path = tmp_path / "qoe_edge_only.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "edge_inference_ms", "wireless_delay_ms", "A_x", "T_i_ms", "correct"])
        w.writeheader()
        for _ in range(50):
            w.writerow({"path": "IoT->Edge", "edge_inference_ms": 50, "wireless_delay_ms": 3, "A_x": 0.8,
                        "T_i_ms": 1000, "correct": 1})
    res = ms.sweep(ms.load_rows(path), [0.5, 2, 10, 100], "ES", trials=300, model="mm1", arrival_rate=1.0, seed=0)
    assert res[0]["mean_Q"] == 0 and res[0]["discard_rate"] == 1
    qs = [r["mean_Q"] for r in res[1:]]
    assert qs == sorted(qs)
