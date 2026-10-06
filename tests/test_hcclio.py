import math
import socket

import numpy as np
import pytest

from hcclio.channel import GammaBackhaul, RayleighChannel, p_los, pathloss_los_db, pathloss_nlos_db
from hcclio.config import load_config
from hcclio.dataset_loader import DatasetLoader, make_synthetic_dataset
from hcclio.e2lm import E2LMServer, probe
from hcclio.iot import IoTDevice
from hcclio.models import StubClassifier, jpeg_ground_truth
from hcclio.mu_sweep import load_rows, sweep
from hcclio.qoe import e2e_latency, ensemble, qoe
from hcclio.servers import CloudServer, EdgeServer
from hcclio.transport import InProcBroker, InProcTransport, pack, unpack


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cfg():
    return load_config(overrides={"models": {"backend": "stub", "stub": {"latency_ms": {"iot": 0, "edge": 0, "cloud": 0}}}})


def test_weights_sum_and_direct_renormalisation(cfg):
    w = cfg["weights"]
    assert w["edge"] == {"w_i": 0.44, "w_ES": 0.56}
    assert math.isclose(w["cloud_direct"]["w_i"], 0.28 / 0.66)
    assert math.isclose(sum(w["cloud_direct"].values()), 1.0)
    with pytest.raises(ValueError):
        load_config(overrides={"weights": {"edge": {"w_i": 0.5, "w_ES": 0.6}}})


def test_pathloss_formulas():
    assert math.isclose(pathloss_los_db(3.5e9, 10), 31.84 + 19 * math.log10(3.5) + 20)
    assert math.isclose(pathloss_nlos_db(3.5e9, 10), 32.4 + 20 * math.log10(3.5) + 30)
    assert math.isclose(p_los(10, 0.5, 18), 0.5 ** (10 / 18))


def test_rayleigh_delay_matches_shannon(cfg):
    ch = RayleighChannel(cfg["wireless"], np.random.default_rng(1))
    s = ch.sample()
    assert math.isclose(s["tx_delay_ms"], 1e6 / s["rate_bps"] * 1000)
    assert math.isclose(s["rate_bps"], 20e6 * math.log2(1 + 10 ** (s["snr_db"] / 10)), rel_tol=1e-9)
    d = np.array([ch.delay_ms() for _ in range(5000)])
    assert (d > 0).all() and np.median(d) < 20  # ~2.4 ms at 10 m LOS, heavy tail in fades


def test_backhaul_gamma(cfg):
    bh = GammaBackhaul(cfg["backhaul"], np.random.default_rng(2))
    m, th = bh.params()
    assert m == math.floor((1 + 1.28 * 4) * 1.5 + 2 * 1.0) == 11
    assert math.isclose(th, 0.2 + 1e6 * 1.5e-6)
    samples = np.array([bh.delay_ms() for _ in range(20000)])
    assert abs(samples.mean() - m * th) / (m * th) < 0.02


def test_qoe_and_discard():
    assert qoe(250, 1000, 0.8) == (pytest.approx(0.6), False)
    assert qoe(1000, 1000, 0.8) == (0.0, True)
    assert e2e_latency({"local_inference_ms": 10, "wireless_delay_ms": 2.5, "cloud_inference_ms": None}) == 12.5


def test_ensemble():
    p1 = np.array([0.7, 0.3])
    p2 = np.array([0.2, 0.8])
    p, c, y = ensemble([p1, p2], [0.44, 0.56])
    assert y == 1 and math.isclose(c, 0.44 * 0.3 + 0.56 * 0.8)


def test_msgpack_numpy_roundtrip():
    a = np.random.default_rng(0).random(1000).astype(np.float32)
    out = unpack(pack({"p": a, "jpeg": b"\xff\xd8x", "t": {"x": 1.5}}))
    assert np.array_equal(out["p"], a) and out["jpeg"] == b"\xff\xd8x" and out["t"]["x"] == 1.5


def test_e2lm_probe():
    port = free_port()
    srv = E2LMServer("127.0.0.1", port, work_iters=1000)
    srv.start_background()
    try:
        assert 0 < probe("127.0.0.1", port, n_probes=3) < 500
    finally:
        srv.shutdown()
        srv.server_close()
    assert probe("127.0.0.1", free_port(), timeout_s=0.5) == 500.0  # unreachable -> timeout


def test_stub_is_deterministic_and_reads_gt(tmp_path):
    make_synthetic_dataset(tmp_path, 5)
    f = next(iter(DatasetLoader(tmp_path)))
    assert jpeg_ground_truth(f.jpeg) == f.ground_truth_idx
    s = StubClassifier("iot", 0.75)
    p1, _ = s.predict(f.jpeg)
    p2, _ = s.predict(f.jpeg)
    assert np.array_equal(p1, p2) and math.isclose(float(p1.sum()), 1.0, rel_tol=1e-5)


def _system(cfg, tmp_path, n=60, **over):
    from hcclio.config import deep_update

    cfg = type(cfg)(deep_update(cfg, over))
    cfg["network"]["edge_host"] = cfg["network"]["cloud_host"] = "127.0.0.1"
    ep, cp = free_port(), free_port()
    cfg["network"]["e2lm"].update({"edge_port": ep, "cloud_port": cp, "server_work_iters": 0})
    broker = InProcBroker()
    edge = EdgeServer(cfg, InProcTransport(broker), start_e2lm=True)
    cloud = CloudServer(cfg, InProcTransport(broker), start_e2lm=True)
    make_synthetic_dataset(tmp_path / "ds", n)
    return cfg, broker, DatasetLoader(tmp_path / "ds"), (edge, cloud)


@pytest.mark.parametrize("strategy", ["hcclio", "local_only", "edge_only", "cloud_only", "dci"])
def test_strategies_end_to_end(cfg, tmp_path, strategy):
    cfg, broker, loader, servers = _system(cfg, tmp_path)
    dev = IoTDevice(cfg, InProcTransport(broker), strategy)
    rows = dev.run(loader, tmp_path / "out.csv", progress_every=0)
    for s in servers:
        s.close()
    assert len(rows) == len(loader)
    tiers = {r["tier"] for r in rows}
    assert "Timeout" not in tiers
    for r in rows:
        assert r["e2e_latency_ms"] == pytest.approx(e2e_latency(r))
        if r["tier"] in ("IoT", "Fallback-IoT"):
            assert r["prediction_idx"] == np.argmax(dev.model.predict(next(f for f in loader if f.frame_id == r["frame_id"]).jpeg)[0])
    if strategy == "local_only":
        assert tiers == {"IoT"}
    if strategy == "edge_only":
        assert tiers == {"Edge"} and all("local_inference_ms" not in r for r in rows)
    if strategy == "cloud_only":
        assert tiers == {"Cloud"} and all(r["path"] == "IoT->Cloud" for r in rows)
    if strategy == "hcclio":
        assert all(r["iot_confidence"] >= 0.8 for r in rows if r["tier"] == "IoT")
        assert all(r["iot_confidence"] < 0.8 for r in rows if r["tier"] != "IoT")
        assert all(r["aggregated_conf"] >= 0.8 for r in rows if r["tier"] in ("Edge", "Cloud"))


def test_latency_gate_fail_goes_direct_to_cloud(cfg, tmp_path):
    cfg, broker, loader, servers = _system(cfg, tmp_path, gates={"tau_lat_ms": 0.0})
    rows = IoTDevice(cfg, InProcTransport(broker), "hcclio").run(loader, tmp_path / "o.csv", progress_every=0)
    for s in servers:
        s.close()
    off = [r for r in rows if r["tier"] != "IoT"]
    assert off and all(r["path"] == "IoT->Cloud" and "lat_fail" in r["route"] for r in off)
    assert all("edge_inference_ms" not in r and r["backhaul_delay_ms"] > 0 for r in off)
    # DCI ignores the latency gate
    rows = IoTDevice(cfg, InProcTransport(broker), "dci").run(loader, tmp_path / "d.csv", progress_every=0)
    assert all(r["path"].startswith("IoT->Edge") for r in rows if r["tier"] != "IoT")


def test_discard_when_slower_than_t_i(cfg, tmp_path):
    cfg, broker, loader, servers = _system(cfg, tmp_path, n=20, qoe={"t_i_ms": 1e-6})
    rows = IoTDevice(cfg, InProcTransport(broker), "hcclio").run(loader, tmp_path / "o.csv", progress_every=0)
    for s in servers:
        s.close()
    assert {r["tier"] for r in rows} == {"Discard"} and all(r["Q_x"] == 0 for r in rows)


def test_mu_sweep_monotone(cfg, tmp_path):
    cfg, broker, loader, servers = _system(cfg, tmp_path)
    IoTDevice(cfg, InProcTransport(broker), "edge_only").run(loader, tmp_path / "e.csv", progress_every=0)
    for s in servers:
        s.close()
    rows = load_rows(tmp_path / "e.csv")
    res = sweep(rows, [0.5, 2, 10, 100], "ES", trials=300, model="mm1", arrival_rate=1.0, seed=0)
    assert res[0]["mean_Q"] == 0 and res[0]["discard_rate"] == 1  # mu <= lambda: unstable queue
    qs = [r["mean_Q"] for r in res[1:]]
    assert qs == sorted(qs)


def test_timm_classifier_path(monkeypatch):
    """Real ViT code path with random weights (no hub download)."""
    timm = pytest.importorskip("timm")
    pytest.importorskip("torch")
    from hcclio.classes import COBOT_CLASSES
    from hcclio.models import TimmClassifier, _blank_jpeg

    create = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained=False, **kw: create(name, pretrained=False, **kw))
    m = TimmClassifier("iot", "vit_small_patch16_224.augreg_in21k_ft_in1k",
                       restrict_classes=list(COBOT_CLASSES.values()))
    p, ms = m.predict(_blank_jpeg())
    assert p.shape == (1000,) and math.isclose(float(p.sum()), 1.0, rel_tol=1e-4) and ms > 0
    assert int(p.argmax()) in COBOT_CLASSES.values()
    assert float(p[[i for i in range(1000) if i not in COBOT_CLASSES.values()]].sum()) == 0.0


def test_class_indices_match_timm():
    pytest.importorskip("timm")
    from timm.data import ImageNetInfo

    from hcclio.classes import COBOT_CLASSES

    info = ImageNetInfo()
    for name, idx in COBOT_CLASSES.items():
        assert name.split("_")[0] in info.index_to_description(idx).lower().replace(" ", "_")


# ---------------------------------------------------------------- IoT tier scripts
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

IOT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "iot_device_tier"
EDGE_DIR = IOT_DIR.parent / "edge_server_tier"
CLOUD_DIR = IOT_DIR.parent / "cloud_server_tier"


def _iot_module(name, folder=None):
    folder = folder or IOT_DIR
    spec = importlib.util.spec_from_file_location(f"{folder.name}_{name}", folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def test_iot_rayleigh_matches_library(cfg):
    er = _iot_module("edge_rayleigh_delay", EDGE_DIR)
    a = er.EdgeRayleighDelay(cfg["wireless"], seed=5)
    b = RayleighChannel(cfg["wireless"], np.random.default_rng(5))
    assert [a.delay_ms() for _ in range(200)] == pytest.approx([b.delay_ms() for _ in range(200)])


def test_iot_qoe_mm1_model():
    q = _iot_module("qoe")
    assert q.qoe(250, 1000, 0.8) == (pytest.approx(0.6), False)
    assert q.qoe(1200, 1000, 0.8) == (0.0, True)
    mu = q.service_time_s(q.complexity_from_model("i", 1e6), 1e6, q.rate_from_measurement("i", 200.0))
    assert mu == pytest.approx(0.2)                                  # mu_i reproduces the measured 200 ms
    assert q.mean_system_time_s(0.2, 1.0, 2.0) == pytest.approx(1 / (5 - 2))
    assert math.isinf(q.mean_system_time_s(0.2, 1.0, 5.0))           # 1/mu <= P lambda -> unstable


def test_iot_load_and_results(tmp_path):
    make_synthetic_dataset(tmp_path / "ds", 4)
    frames = _iot_module("load_dataset").load_dataset(str(tmp_path / "ds"))
    assert len(frames) == 4 and frames[0].jpeg[:2] == b"\xff\xd8"
    res = _iot_module("results")
    with res.ResultsCSV(tmp_path / "r.csv") as w:
        w.save({"frame_id": 0, "tier": "IoT", "correct": 1, "Q_x": 0.5, "e2e_latency_ms": 10.0})
    assert "1 frames" in res.summarise(tmp_path / "r.csv")


@pytest.mark.skipif(shutil.which("mosquitto") is None, reason="needs a local mosquitto broker")
def test_iot_main_end_to_end(cfg, tmp_path):
    import yaml

    port = free_port()
    broker = subprocess.Popen(["mosquitto", "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = dict(cfg)
        c.pop("_path", None)
        c["network"] = {**c["network"], "edge_host": "127.0.0.1", "cloud_host": "127.0.0.1"}
        c["network"]["mqtt"] = {**c["network"]["mqtt"], "broker_host": "127.0.0.1", "broker_port": port}
        c["network"]["e2lm"] = {**c["network"]["e2lm"], "edge_port": free_port(), "cloud_port": free_port()}
        c["weights"] = {k: v for k, v in c["weights"].items() if k != "cloud_direct"}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(c))
        cfg2 = load_config(path)
        from hcclio.transport import make_transport

        import time
        time.sleep(0.5)
        # all three tiers run as their own main.py
        tiers = [subprocess.Popen([sys.executable, str(d / "main.py"), "--config", str(path), "--backend", "stub"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for d in (EDGE_DIR, CLOUD_DIR)]
        for e2lm_port in (cfg2["network"]["e2lm"]["edge_port"], cfg2["network"]["e2lm"]["cloud_port"]):
            for _ in range(100):
                with socket.socket() as s_:
                    if s_.connect_ex(("127.0.0.1", e2lm_port)) == 0:
                        break
                time.sleep(0.1)
        time.sleep(0.5)  # let both MQTT clients subscribe
        make_synthetic_dataset(tmp_path / "ds", 30)
        out = tmp_path / "q.csv"
        r = subprocess.run([sys.executable, str(IOT_DIR / "main.py"), "--config", str(path), "--backend", "stub",
                            "--dataset", str(tmp_path / "ds"), "--csv", str(out)], capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        import csv as _csv

        rows = list(_csv.DictReader(open(out)))
        assert len(rows) == 30 and {x["tier"] for x in rows} <= {"IoT", "Edge", "Cloud", "Fallback-IoT"}
        off = [x for x in rows if x["tier"] != "IoT"]
        assert off and all(float(x["wireless_delay_ms"]) > 0 for x in off)  # Rayleigh delay came from the Edge
        assert any("Edge" in x["path"] for x in off)
        cloud_rows = [x for x in rows if "Cloud" in x["path"]]
        assert all(float(x["backhaul_delay_ms"]) > 0 for x in cloud_rows)  # added by the Cloud
        assert all(not x["backhaul_delay_ms"] for x in rows if "Cloud" not in x["path"])
        for t in tiers:
            t.terminate()
    finally:
        broker.terminate()
