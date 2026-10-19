"""Scenario loader (Spec 1.5). `gold` is returned separately: only eval/ and fit() may touch it."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from csmr.types import Memory, Observation


@dataclass
class GoldLabels:
    actions: dict[str, str]
    dependencies: list[tuple[str, str]]
    eval_questions: list[dict]


@dataclass
class Scenario:
    id: str
    memories: list[Memory]
    observation: Observation
    gold: Optional[GoldLabels] = None
    meta: dict = field(default_factory=dict)  # category, domain, note


def load_scenario(path: str | Path) -> Scenario:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    g = d.get("gold")
    gold = GoldLabels(g["actions"], [tuple(p) for p in g["dependencies"]], g["eval_questions"]) if g else None
    return Scenario(d["id"], [Memory(**m) for m in d["memories"]], Observation(**d["observation"]), gold,
                    {k: d[k] for k in ("category", "domain", "note") if k in d})
