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


def _resolve(path: str) -> Path:
    """As given if it exists (relative to the cwd), else relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def load_overlay(path: str) -> dict:
    """A config overlay file (e.g. config/tuned.yaml): a mapping of keys to put on top of config.yaml."""
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"config overlay not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config overlay {path} must be a mapping, got {type(data).__name__}")
    return data


def _merge(base: dict, top: dict) -> None:
    """Deep merge in place: nested mappings merge key by key, anything else replaces."""
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


@lru_cache(maxsize=None)
def _load(path: str, overrides: tuple[str, ...], overlay: str | None) -> dict:
    cfg = yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))
    if overlay:
        _merge(cfg, load_overlay(overlay))
    for item in overrides:  # --set wins over the overlay
        key, raw = item.split("=", 1)
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)  # "0.3" -> float, "false" -> bool
    return cfg


def load_config(path: str = "config.yaml", overrides: list[str] | None = None, overlay: str | None = None) -> dict:
    """config.yaml, then the overlay file (if any), then `--set` overrides. Cached; returns a deep copy so callers
    can't corrupt the cache."""
    return copy.deepcopy(_load(str(path), parse_overrides(overrides), overlay))
