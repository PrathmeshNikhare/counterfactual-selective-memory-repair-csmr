"""Phase B — candidate construction (Paper IV-B). Permissive by design: entering C is not repair."""
from __future__ import annotations

from csmr import models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.types import CandidateInfo, Observation, StateChange, same_entity


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

    def related(m, anchor_id: str) -> bool:
        a = mems[anchor_id]
        return same_entity(m.entity, a.entity) and (m.attribute == a.attribute or cos(vecs[m.id], vecs[a.id]) >= cfg["hop_sim"])

    cands = {r: CandidateInfo(r, 0, "root", is_root=True) for r in roots if r in mems}

    frontier = [i for i, c in cands.items() if c.is_root]  # hops expand from roots only (SPEC §3)
    for hop in range(1, cfg["max_hops"] + 1):
        new = [i for i in ids if i not in cands and any(related(mems[i], a) for a in frontier)]
        for i in new:
            cands[i] = CandidateInfo(i, hop, "hop")
        frontier = new

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
