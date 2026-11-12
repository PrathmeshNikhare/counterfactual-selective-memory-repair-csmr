"""Phase B — candidate construction (Paper IV-B). Permissive by design: entering C is not repair."""
from __future__ import annotations

import numpy as np

from csmr import models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.types import CandidateInfo, Memory, Observation, StateChange, same_entity


def hop_expansion(roots: list[str], mems: dict[str, Memory], vecs: dict[str, np.ndarray], hops: int,
                  hop_sim: float) -> dict[str, int]:
    """The local memory graph of Paper IV-B: m and a are linked iff same entity and (same attribute or
    cosine >= hop_sim). Breadth-first from the roots: memory id -> hop (roots 0), up to `hops`, in store order.
    Shared by Phase B and the graph_propagation baseline."""
    def linked(m: Memory, a: Memory) -> bool:
        return same_entity(m.entity, a.entity) and (
            m.attribute == a.attribute or float(vecs[m.id] @ vecs[a.id]) >= hop_sim)

    reached = {r: 0 for r in roots if r in mems}
    frontier = list(reached)
    for hop in range(1, hops + 1):
        new = [i for i in mems if i not in reached and any(linked(mems[i], mems[a]) for a in frontier)]
        reached.update((i, hop) for i in new)
        frontier = new
    return reached


def build_candidates(delta: StateChange, obs: Observation, roots: list[str],
                     store: MemoryStoreAdapter, cfg: dict | None = None) -> dict[str, CandidateInfo]:
    """Paper IV-B: roots ∪ (top-K by cosine, entity/attribute-filtered) ∪ 1..max_hops expansion from roots."""
    cfg = cfg or load_config()
    mems = {m.id: m for m in store.all()}
    ids = list(mems)
    if not ids:
        return {}
    vecs = dict(zip(ids, models.embed([mems[i].text for i in ids])))
    q = models.embed([obs.text])[0]

    def cos(a, b) -> float:
        return float(a @ b)

    # hops expand from roots only (SPEC §3)
    cands = {i: CandidateInfo(i, h, "root", is_root=True) if h == 0 else CandidateInfo(i, h, "hop")
             for i, h in hop_expansion(roots, mems, vecs, cfg["max_hops"], cfg["hop_sim"]).items()}

    # top-K hits not already reached by hop expansion (hop-reached ones keep their hop, SPEC §3)
    # entity restriction falls back to all memories if it leaves fewer than K
    pool = [i for i in ids if same_entity(mems[i].entity, delta.entity)]
    if len(pool) < cfg["top_k"]:
        pool = ids
    for i in sorted(pool, key=lambda i: -cos(q, vecs[i]))[:cfg["top_k"]]:
        m = mems[i]
        # attribute/similarity filter: same entity and (same attribute as Δ or similar enough to the observation)
        if i not in cands and same_entity(m.entity, delta.entity) and (m.attribute == delta.attribute or cos(q, vecs[i]) >= cfg["topk_min_sim"]):
            cands[i] = CandidateInfo(i, 2, "topk")
    return cands
