"""Loads config.yaml (CLAUDE.md rule 6: all tunables live there). Spec 1.1."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def parse_overrides(items: list[str] | None) -> tuple[str, ...]:
    """Validate CLI `--set key.sub=value` items; returns a hashable tuple."""
    items = tuple(items or ())
    for s in items:
        if "=" not in s:
            raise ValueError(f"override must look like key=value, got {s!r}")
    return items


@lru_cache(maxsize=None)
def _load(path: str, overrides: tuple[str, ...]) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = p if p.exists() else ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    for item in overrides:
        key, raw = item.split("=", 1)
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)  # "0.3" -> float, "false" -> bool
    return cfg


def load_config(path: str = "config.yaml", overrides: list[str] | None = None) -> dict:
    """Cached; returns a deep copy so callers can't corrupt the cache."""
    return copy.deepcopy(_load(str(path), parse_overrides(overrides)))
