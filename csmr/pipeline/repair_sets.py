"""Candidate repair sets R0..R4 and feasibility (Paper IV-D, Section V)."""
from __future__ import annotations

from csmr.types import MODIFYING_ACTIONS, Action, CandidateInfo, RepairSet


def generate_repair_sets(cands: dict[str, CandidateInfo], proposed: dict[str, Action], cfg: dict) -> list[RepairSet]:
    """R0 none; R1 roots; R2 +hop-1; R3 +hop>=2 (all proposed non-KEEP); R4 broad (P_dep >= r4_min_pdep).

    Deduplicated by content, keeping the lowest-indexed name. Infeasible sets are kept but marked.
    """
    def act(mid: str) -> Action:
        return proposed.get(mid, Action.KEEP)

    r1 = {m: (act(m) if act(m) != Action.KEEP else Action.REVISE) for m, c in cands.items() if c.is_root}
    r2 = {**r1, **{m: act(m) for m, c in cands.items() if c.hop == 1 and not c.is_root and act(m) != Action.KEEP}}
    r3 = {**r2, **{m: act(m) for m, c in cands.items() if c.hop >= 2 and not c.is_root and act(m) != Action.KEEP}}
    r4 = {m: (act(m) if act(m) in MODIFYING_ACTIONS else Action.REVISE)
          for m, c in cands.items() if c.is_root or c.p_dep >= cfg["r4_min_pdep"]}

    out: list[RepairSet] = []
    seen: set[tuple] = set()
    for name, actions in zip(("R0", "R1", "R2", "R3", "R4"), ({}, r1, r2, r3, r4)):
        rs = RepairSet(name=name, actions=dict(actions))
        if rs.key() in seen:
            continue
        seen.add(rs.key())
        rs.feasible = is_feasible(rs, cands, cfg)
        out.append(rs)
    return out


def is_feasible(rs: RepairSet, cands: dict[str, CandidateInfo], cfg: dict) -> bool:
    """Hard rules (Section V): actions only on members of C; REVISE/REVOKE need S >= tau_e. Sets reason on failure."""
    rs.infeasible_reason = None
    for mid, a in rs.actions.items():
        if a == Action.KEEP:
            continue
        if mid not in cands:
            rs.infeasible_reason = f"{mid} not in candidate set"
        elif a in MODIFYING_ACTIONS and cands[mid].s_evidence < cfg["tau_e"]:
            rs.infeasible_reason = f"{mid} {a.value} but S={cands[mid].s_evidence:.2f} < tau_e"
        else:
            continue
        rs.feasible = False
        return False
    rs.feasible = True
    return True
