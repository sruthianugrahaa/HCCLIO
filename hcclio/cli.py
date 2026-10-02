"""Command-line entry points shared by the scripts under outputs/."""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from .config import load_config, resolve_path

CSV_NAMES = {
    "hcclio": None,  # logging.hcclio_csv (qoe_coclio.csv)
    "local_only": "qoe_local_only.csv",
    "edge_only": "qoe_edge_only.csv",
    "cloud_only": "qoe_cloud_only.csv",
    "dci": "qoe_distributed_benchmark.csv",
}


def _setup_logging(verbose: bool = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


def csv_path_for(cfg, strategy: str):
    name = CSV_NAMES[strategy] or cfg["logging"]["hcclio_csv"]
    return resolve_path(cfg["logging"]["log_dir"]) / name


def _common(ap: argparse.ArgumentParser):
    ap.add_argument("--config", default=None, help="path to hcclio.yaml (default config/hcclio.yaml)")
    ap.add_argument("--backend", choices=["timm", "stub"], default=None, help="override models.backend")
    ap.add_argument("-v", "--verbose", action="store_true")


def _cfg(args):
    over = {"models": {"backend": args.backend}} if getattr(args, "backend", None) else None
    return load_config(args.config, over)


def _wait_forever():
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:
            pass
    while not stop.wait(1.0):
        pass


def iot_main(argv=None, default_strategy: str = "hcclio"):
    from .dataset_loader import DatasetLoader
    from .iot import STRATEGIES, IoTDevice
    from .transport import make_transport

    ap = argparse.ArgumentParser(description="IoT device tier (Raspberry Pi 5): run a strategy over the dataset")
    _common(ap)
    ap.add_argument("--strategy", choices=STRATEGIES, default=default_strategy)
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames")
    ap.add_argument("--dataset", default=None, help="dataset root (default dataset.root)")
    ap.add_argument("--csv", default=None, help="output CSV (default outputs/logs/qoe_<strategy>.csv)")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = _cfg(args)
    d = cfg["dataset"]
    loader = DatasetLoader(args.dataset or d["root"], d["manifest"], args.limit or d["n_frames"], d["jpeg_quality"])
    tx = None if args.strategy == "local_only" else make_transport(cfg, f"iot-{args.strategy}")
    dev = IoTDevice(cfg, tx, args.strategy)
    out = args.csv or csv_path_for(cfg, args.strategy)
    rows = dev.run(loader, out)
    logging.info("wrote %d rows to %s", len(rows), out)
    if tx:
        tx.close()


def edge_main(argv=None):
    from .servers import EdgeServer
    from .transport import make_transport

    ap = argparse.ArgumentParser(description="Edge server tier (ViT-Base); also hosts the E2LM probe server")
    _common(ap)
    ap.add_argument("--no-e2lm", action="store_true", help="do not start the E2LM probe server")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = _cfg(args)
    srv = EdgeServer(cfg, make_transport(cfg, "edge"), start_e2lm=not args.no_e2lm)
    logging.info("Edge ready: %s on %s", cfg["models"]["edge"]["name"], cfg["network"]["mqtt"]["broker_host"])
    _wait_forever()
    srv.close()


def cloud_main(argv=None):
    from .servers import CloudServer
    from .transport import make_transport

    ap = argparse.ArgumentParser(description="Cloud server tier (ViT-Large); also hosts the E2LM probe server")
    _common(ap)
    ap.add_argument("--no-e2lm", action="store_true")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = _cfg(args)
    srv = CloudServer(cfg, make_transport(cfg, "cloud"), start_e2lm=not args.no_e2lm)
    logging.info("Cloud ready: %s via broker %s", cfg["models"]["cloud"]["name"], cfg["network"]["mqtt"]["broker_host"])
    _wait_forever()
    srv.close()
