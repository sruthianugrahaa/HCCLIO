"""Load and validate the shared YAML config."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "hcclio.yaml"


class Config(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value


def _check_sum(name: str, weights: dict) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights.{name} must sum to 1, got {total:.6f} ({weights})")


def resolve_path(p: str | os.PathLike) -> Path:
    """Expand ~ and make repo-relative paths absolute."""
    path = Path(os.path.expanduser(str(p)))
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("HCCLIO_CONFIG", DEFAULT_CONFIG))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)

    w = cfg["weights"]
    if not w.get("cloud_direct"):
        s = w["cloud"]["w_i"] + w["cloud"]["w_CS"]
        w["cloud_direct"] = {"w_i": w["cloud"]["w_i"] / s, "w_CS": w["cloud"]["w_CS"] / s}
    for tier in ("edge", "cloud", "cloud_direct"):
        _check_sum(tier, w[tier])
    cfg["_path"] = str(path)
    return Config(cfg)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
