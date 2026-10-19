"""Store interface. CSMR never talks to a database directly; every backend implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from csmr.types import Memory


class MemoryStoreAdapter(ABC):
    @abstractmethod
    def all(self, entity: Optional[str] = None, include_revoked: bool = False) -> list[Memory]:
        """All memories (optionally one entity). Revoked excluded unless asked."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory: ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[Memory]:
        """Top-k active memories by embedding similarity."""

    @abstractmethod
    def add(self, memory: Memory) -> None: ...

    @abstractmethod
    def update(self, memory_id: str, *, text: Optional[str] = None, status: Optional[str] = None,
               condition: Optional[str] = None, clear_condition: bool = False, reason: str = "") -> Memory:
        """MUST push the previous version into history and increment version. Never delete.
        clear_condition=True resets condition to None (mutually exclusive with `condition`)."""

    @abstractmethod
    def snapshot(self) -> "MemoryStoreAdapter":
        """Independent deep copy (used by tests / baselines, not by the overlay)."""
