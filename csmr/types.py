"""Shared types for CSMR. This file is the contract between all modules.
Do not change fields without updating every user (see CLAUDE.md rule 8)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    """Paper IV-F repair actions."""
    KEEP = "KEEP"
    VERIFY = "VERIFY"
    CONDITIONAL = "CONDITIONAL"
    REVISE = "REVISE"
    REVOKE = "REVOKE"


MODIFYING_ACTIONS = {Action.REVISE, Action.REVOKE}


@dataclass
class Memory:
    id: str
    text: str
    entity: str
    attribute: str
    time: str                      # ISO 8601
    source: str = "user"           # user | tool | third_party | inferred
    status: str = "active"         # active | revoked | conditional | pending_verify
    version: int = 1
    condition: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)  # previous versions, never deleted

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    text: str
    source: str = "user"
    time: str = ""


@dataclass
class StateChange:
    """Paper IV-A: Δ = (entity, attribute, old, new, time, source, confidence)."""
    entity: str
    attribute: str
    old: Optional[str]
    new: str
    time: str
    source: str
    confidence: float
    statement: str = ""  # third-person restatement of the observation, e.g. "The user sold their car."
    change_type: str = "replace"  # "remove" (fact no longer holds) | "replace" (fact has a new value)

    def as_sentence(self) -> str:
        return f"The {self.entity}'s {self.attribute.replace('_', ' ')} is now {self.new}."


@dataclass
class CandidateInfo:
    """Everything computed about one candidate memory in Phases B-C."""
    memory_id: str
    hop: int                        # 0 = root, 1, 2 (top-K-only candidates use 2)
    retrieved_by: str               # root | topk | hop
    is_root: bool = False
    dep_features: dict[str, float] = field(default_factory=dict)
    p_dep: float = 0.0              # P(D=1 | z)
    ev_features: dict[str, float] = field(default_factory=dict)
    s_evidence: float = 0.0         # S(E, m)
    hedged: bool = False
    temporary: bool = False
    proposed: Action = Action.KEEP


@dataclass
class RepairSet:
    """Paper IV-D: one alternative R. Memories not in `actions` are KEEP."""
    name: str                       # R0..R4
    actions: dict[str, Action] = field(default_factory=dict)
    feasible: bool = True
    infeasible_reason: Optional[str] = None
    score: dict[str, float] = field(default_factory=dict)  # benefit, collateral, evidence_cost, size, total

    def size(self) -> int:
        return sum(1 for a in self.actions.values() if a != Action.KEEP)

    def key(self) -> tuple:
        return tuple(sorted((k, v.value) for k, v in self.actions.items() if v != Action.KEEP))


@dataclass
class Trace:
    """Full audit log of one observe() call (paper IV-E)."""
    observation: dict
    delta: Optional[dict] = None
    note: Optional[dict] = None     # {entity, attribute, statement} of the observation (Spec 2), used when Δ is None
    roots: list[str] = field(default_factory=list)
    candidates: dict[str, dict] = field(default_factory=dict)
    probes: list[str] = field(default_factory=list)
    repair_sets: list[dict] = field(default_factory=list)
    chosen: Optional[str] = None
    final_actions: dict[str, str] = field(default_factory=dict)
    transitions: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    llm_stats: dict = field(default_factory=dict)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
