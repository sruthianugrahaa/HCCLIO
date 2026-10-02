"""Run all three tiers on one machine (loopback) to check the pipeline end to end.

Edge and Cloud servers, their E2LM probe servers (127.0.0.1:9001/9002) and the IoT
device all run in this process. Messages go through a real MQTT broker if --mqtt is
given, otherwise through an in-process broker. Every strategy is run, then the
report and the mu sweep are produced.

    python scripts/loopback.py                              # stub models, synthetic frames
    python scripts/loopback.py --mqtt 127.0.0.1:1883        # through a local mosquitto
    python scripts/loopback.py --backend timm --dataset ~/testbed/dataset/imagenet_1000 --limit 50
"""
import argparse
import logging
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcclio.config import load_config  # noqa: E402
from hcclio.dataset_loader import DatasetLoader, make_synthetic_dataset  # noqa: E402
from hcclio.iot import STRATEGIES, IoTDevice  # noqa: E402
from hcclio.mu_sweep import STRATEGY_LABELS, load_rows, plot, sweep, write_csv  # noqa: E402
from hcclio.report import make_report  # noqa: E402
from hcclio.servers import CloudServer, EdgeServer  # noqa: E402
from hcclio.transport import InProcBroker, make_transport  # noqa: E402

CSV = {"hcclio": "qoe_coclio.csv", "local_only": "qoe_local_only.csv", "edge_only": "qoe_edge_only.csv",
       "cloud_only": "qoe_cloud_only.csv", "dci": "qoe_distributed_benchmark.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["stub", "timm"], default="stub")
    ap.add_argument("--mqtt", default=None, help="host:port of a broker (default: in-process)")
    ap.add_argument("--dataset", default=None, help="dataset root (default: synthetic frames)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="writes <out>/logs and <out>/reports")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    over = {"models": {"backend": args.backend},
            "network": {"edge_host": "127.0.0.1", "cloud_host": "127.0.0.1",
                        "e2lm": {"edge_port": 9001, "cloud_port": 9002},
                        "mqtt": {"response_timeout_s": 120}}}
    if args.mqtt:
        host, port = args.mqtt.split(":")
        over["network"]["mqtt"].update({"broker_host": host, "broker_port": int(port)})
    cfg = load_config(args.config, over)

    root = args.dataset
    if root is None:
        root = tempfile.mkdtemp(prefix="hcclio_synth_")
        make_synthetic_dataset(root, args.limit, seed=cfg["sim"]["seed"])
    loader = DatasetLoader(root, "manifest.csv", args.limit, cfg["dataset"]["jpeg_quality"])

    broker = None if args.mqtt else InProcBroker()
    edge = EdgeServer(cfg, make_transport(cfg, "edge", broker), start_e2lm=True)
    cloud = CloudServer(cfg, make_transport(cfg, "cloud", broker), start_e2lm=True)

    out = pathlib.Path(args.out)
    for strat in args.strategies:
        tx = None if strat == "local_only" else make_transport(cfg, f"iot-{strat}", broker)
        dev = IoTDevice(cfg, tx, strat)
        dev.run(loader, out / "logs" / CSV[strat], progress_every=0)
        if tx:
            tx.close()
        logging.info("%s done", strat)

    summaries = make_report(out / "logs", out / "reports")
    print((out / "reports" / "summary.md").read_text())
    s = cfg["mu_sweep"]
    per = {}
    flat = []
    for strat in args.strategies:
        f = out / "logs" / CSV[strat]
        res = sweep(load_rows(f), s["mu_values"], s["tier"], 200, s["service_model"], float(s["arrival_rate"]), 0)
        per[STRATEGY_LABELS[f.name]] = res
        flat += [{"strategy": STRATEGY_LABELS[f.name], **r} for r in res]
    write_csv(flat, out / "reports" / f"mu_sweep_{s['tier']}.csv")
    plot(per, s["tier"], out / "reports" / f"mu_sweep_{s['tier']}.png")
    edge.close()
    cloud.close()
    return summaries


if __name__ == "__main__":
    main()
