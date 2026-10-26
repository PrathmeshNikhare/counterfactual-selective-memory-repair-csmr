"""In-memory / JSON store (Spec 1.4). Versioned; never deletes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np

from csmr import models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.types import Memory, same_entity


class InMemoryAdapter(MemoryStoreAdapter):
    def __init__(self, memories: list[Memory]):
        self._m: dict[str, Memory] = {m.id: m for m in memories}

    def all(self, entity: Optional[str] = None, include_revoked: bool = False) -> list[Memory]:
        return [m for m in self._m.values()
                if (entity is None or same_entity(m.entity, entity)) and (include_revoked or m.status != "revoked")]

    def get(self, memory_id: str) -> Memory:
        return self._m[memory_id]

    def search(self, query: str, k: int) -> list[Memory]:
        ms = self.all()
        if not ms:
            return []
        sims = models.embed([m.text for m in ms]) @ models.embed([query])[0]
        return [ms[i] for i in np.argsort(-sims)[:k]]

    def add(self, memory: Memory) -> None:
        if memory.id in self._m:
            raise ValueError(f"duplicate memory id {memory.id}")
        self._m[memory.id] = memory

    def update(self, memory_id: str, *, text: Optional[str] = None, status: Optional[str] = None,
               condition: Optional[str] = None, clear_condition: bool = False, reason: str = "") -> Memory:
        if clear_condition and condition is not None:
            raise ValueError("condition and clear_condition are mutually exclusive")
        m = self._m[memory_id]
        prev = {k: v for k, v in m.to_dict().items() if k != "history"}
        prev["reason"] = reason
        m.history.append(prev)
        m.version += 1
        if text is not None:
            m.text = text
        if status is not None:
            m.status = status
        if clear_condition:
            m.condition = None
        elif condition is not None:
            m.condition = condition
        return m

    def snapshot(self) -> "InMemoryAdapter":
        return InMemoryAdapter(copy.deepcopy(list(self._m.values())))

    @classmethod
    def from_json(cls, path: str | Path) -> "InMemoryAdapter":
        return cls([Memory(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([m.to_dict() for m in self._m.values()], indent=2), encoding="utf-8")
