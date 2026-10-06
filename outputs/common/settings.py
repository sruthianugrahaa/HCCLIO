"""settings.py - loads common/hcclio.yaml, the one config file every tier reads.

Checks that each tier's ensemble weights sum to 1 and derives the direct
IoT -> Cloud weights (w_i, w_CS renormalised) when the YAML leaves them empty.
Relative paths in the YAML are relative to the outputs/ folder.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

OUTPUTS_DIR = Path(__file__).resolve().parents[1]          # .../outputs
DEFAULT_CONFIG = Path(__file__).resolve().with_name("hcclio.yaml")


def resolve_path(p) -> Path:
    path = Path(os.path.expanduser(str(p)))
    return path if path.is_absolute() else OUTPUTS_DIR / path


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings(path=None, overrides: dict | None = None) -> dict:
    path = Path(path or os.environ.get("HCCLIO_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    w = cfg["weights"]
    if not w.get("cloud_direct"):
        s = w["cloud"]["w_i"] + w["cloud"]["w_CS"]
        w["cloud_direct"] = {"w_i": w["cloud"]["w_i"] / s, "w_CS": w["cloud"]["w_CS"] / s}
    for tier in ("edge", "cloud", "cloud_direct"):
        total = sum(w[tier].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights.{tier} must sum to 1, got {total:.6f}")
    cfg["_path"] = str(path)
    return cfg
