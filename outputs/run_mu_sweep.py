"""Monte-Carlo sweep of mean QoE vs processing rate mu_ES (or mu_CS) on the logged CSVs.

    python outputs/run_mu_sweep.py                      # every outputs/logs/qoe_*.csv
    python outputs/run_mu_sweep.py --csv outputs/logs/qoe_coclio.csv --tier ES --trials 1000

Writes outputs/reports/mu_sweep_<tier>.csv and mu_sweep_<tier>.png.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root

from hcclio.config import load_config, resolve_path  # noqa: E402
from hcclio.mu_sweep import STRATEGY_LABELS, load_rows, plot, sweep, write_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", nargs="*", help="logged CSVs (default: all outputs/logs/qoe_*.csv)")
    ap.add_argument("--tier", choices=["ES", "CS"], default=None)
    ap.add_argument("--mu", type=float, nargs="*", help="mu values in frames/s")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--service-model", choices=["mm1", "exp", "det"], default=None)
    ap.add_argument("--arrival-rate", type=float, default=None)
    ap.add_argument("--t-i-ms", type=float, default=None, help="override the logged T_i^comp")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    s = cfg["mu_sweep"]
    tier = args.tier or s["tier"]
    files = [pathlib.Path(p) for p in args.csv] if args.csv else sorted(resolve_path(cfg["logging"]["log_dir"]).glob("qoe_*.csv"))
    if not files:
        sys.exit("no logged CSVs found - run the IoT tier first")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else resolve_path(cfg["logging"]["report_dir"])

    per_strategy, flat = {}, []
    for f in files:
        label = STRATEGY_LABELS.get(f.name, f.stem)
        res = sweep(load_rows(f), args.mu or s["mu_values"], tier, args.trials or int(s["trials"]),
                    args.service_model or s["service_model"],
                    args.arrival_rate if args.arrival_rate is not None else float(s["arrival_rate"]),
                    int(s["seed"]), args.t_i_ms)
        per_strategy[label] = res
        flat += [{"strategy": label, **r} for r in res]
        best = max(res, key=lambda r: r["mean_Q"])
        print(f"{label:18s} best mean Q = {best['mean_Q']:.4f} at mu_{tier} = {best['mu']:g}")
    write_csv(flat, out_dir / f"mu_sweep_{tier}.csv")
    png = plot(per_strategy, tier, out_dir / f"mu_sweep_{tier}.png")
    print(f"wrote {out_dir / f'mu_sweep_{tier}.csv'}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()
