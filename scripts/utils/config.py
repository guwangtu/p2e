from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge upd into base (returns new dict)."""
    base = deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base
