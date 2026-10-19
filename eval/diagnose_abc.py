"""Diagnose Phases A-C on every dev scenario (real LLM, no stubs). Reads gold for printing only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import dependency, evidence
from csmr.pipeline.actions import propose_action
from csmr.pipeline.candidates import build_candidates
from csmr.pipeline.extract import extract_state_change, find_roots
from csmr.scenario import load_scenario


def _f(d: dict) -> str:
    return ", ".join(f"{k}={v:.2f}" for k, v in d.items()) or "-"


def diagnose(path: Path, cfg: dict) -> tuple[int, int]:
    sc = load_scenario(path)
    store = InMemoryAdapter(sc.memories)
    delta = extract_state_change(sc.observation, [m.attribute for m in sc.memories])
    print(f"\n=== {sc.id}  |  obs: {sc.observation.text!r}")
    if delta is None:
        print("delta: None (no state change extracted)")
        cands, roots = {}, []
    else:
        roots = find_roots(delta, sc.observation, store)
        cands = build_candidates(delta, sc.observation, roots, store)
        dependency.assess_dependency(cands, store, delta, sc.observation)
        evidence.assess_evidence(cands, store, delta, sc.observation)
        for m, i in cands.items():
            i.proposed = propose_action(m, i, cfg, delta.change_type)
    print("| memory | root? | in C? | hop | P_dep | S | hedged | temporary | proposed | gold | match |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    ok = 0
    for m in sc.memories:
        i = cands.get(m.id)
        prop = i.proposed.value if i else "KEEP"
        gold = sc.gold.actions.get(m.id, "?") if sc.gold else "?"
        match = prop in gold.split("/")
        ok += match
        print(f"| {m.id} | {m.id in roots} | {i is not None} | {i.hop if i else '-'} | "
              f"{f'{i.p_dep:.2f}' if i else '-'} | {f'{i.s_evidence:.2f}' if i else '-'} | "
              f"{i.hedged if i else '-'} | {i.temporary if i else '-'} | {prop} | {gold} | {'Y' if match else 'N'} |")
    if delta:
        print(f"delta: entity={delta.entity} attribute={delta.attribute} statement={delta.statement!r} type={delta.change_type} conf={delta.confidence}")
    for m, i in cands.items():
        print(f"  {m}: dep[{_f(i.dep_features)}]\n      ev[{_f(i.ev_features)}]")
    return ok, len(sc.memories)


def main() -> None:
    cfg = load_config()
    summary = {}
    for p in sorted(Path("data/dev").glob("*.json")):
        summary[p.stem] = diagnose(p, cfg)
    print("\n=== SUMMARY (proposed action matches gold)")
    for k, (ok, n) in summary.items():
        print(f"{k}: {ok}/{n}")
    print(f"total: {sum(o for o, _ in summary.values())}/{sum(n for _, n in summary.values())}")


if __name__ == "__main__":
    main()
