"""Presentation report: run CSMR on showcase scenarios and compare answers with / without repair.

    python examples/showcase.py [scenario.json ...]     (default: every file in data/showcase/)

Prints a markdown report per scenario and saves it to results/showcase/<id>.md.
The pipeline sees only memories + observation. Gold is read here solely for the gold-action and gold-answer columns.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csmr import llm  # noqa: E402
from csmr.adapters.memory import InMemoryAdapter  # noqa: E402
from csmr.api import CSMR  # noqa: E402
from csmr.pipeline.agent import answer  # noqa: E402
from csmr.pipeline.overlay import render  # noqa: E402
from csmr.scenario import load_scenario  # noqa: E402

OUT_DIR = Path("results/showcase")
GRID_C, GRID_S, GRID_E = (0.1, 0.25, 0.5), (0.0, 0.01, 0.02, 0.05), (0.0, 0.05, 0.1)  # lambda sensitivity grid


def short(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def cell(s) -> str:
    return " ".join(str(s).split()).replace("|", "\\|")


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]


def agent_view(memories) -> list:
    """Same rendering the pipeline's agent uses (revoked hidden, unverified/conditional tagged)."""
    return [r for m in memories if (r := render(m))]


def pick(sets: list[dict], lc: float, le: float, ls: float, eps: float) -> str:
    """Re-run the selection rule (counterfactual.select) on stored scores under other lambdas; no LLM calls."""
    feas = [r for r in sets if r["feasible"]]
    tot = {r["name"]: (s := r["score"])["benefit"] - lc * s["collateral"] - le * s["evidence_cost"] - ls * s["size"]
           for r in feas}
    top = max(tot.values())
    return min((r for r in feas if tot[r["name"]] >= top - eps), key=lambda r: r["score"]["size"])["name"]


def sensitivity(sets: list[dict], chosen: str, cfg: dict) -> list[str]:
    eps, le0 = cfg["tie_eps"], cfg["lambda_e"]
    if cfg["ablation"]["use_minimality"]:  # self-check: the configured cell reproduces the pipeline's choice
        assert pick(sets, cfg["lambda_c"], le0, cfg["lambda_s"], eps) == chosen
    rows = []
    for lc in GRID_C:
        row = [f"λc={lc}"]
        for ls in GRID_S:
            name = pick(sets, lc, le0, ls, eps)
            row.append(f"**{name}**" if (lc, ls) == (cfg["lambda_c"], cfg["lambda_s"]) else name)
        rows.append(row)
    diffs = [f"λc={lc}, λs={ls}: " + " / ".join(f"{pick(sets, lc, le, ls, eps)} (λe={le})" for le in GRID_E)
             for lc in GRID_C for ls in GRID_S if len({pick(sets, lc, le, ls, eps) for le in GRID_E}) > 1]
    out = ["## Lambda sensitivity", "",
           f"Set that would be chosen under other weights, recomputed offline from the scores above (benefit, collateral, "
           f"evidence cost and size do not depend on λ). λe = {le0} (config); configured cell in bold.", ""]
    out += table(["", *(f"λs={ls}" for ls in GRID_S)], rows)
    out += ["", f"λe ∈ {{{', '.join(map(str, GRID_E))}}}: "
            + (f"changes the choice in {len(diffs)}/{len(GRID_C) * len(GRID_S)} cells — " + "; ".join(diffs)
               if diffs else "never changes the choice.")]
    return out


def report(path: Path) -> tuple[str, str]:
    sc = load_scenario(path)
    sid, meta, gold, obs = sc.id, sc.meta, sc.gold, sc.observation
    before = {m.id: copy.deepcopy(m) for m in sc.memories}  # the store mutates Memory objects in place
    csmr = CSMR(InMemoryAdapter(copy.deepcopy(sc.memories)))
    del sc  # gold stays in `gold` (report columns only); nothing below hands it to the pipeline
    t = csmr.observe(obs.text, obs.source, obs.time)
    store = csmr.store
    gold_actions = gold.actions if gold else {}

    out = [f"# {sid}", "", f"- **Domain:** {meta.get('domain', '-')}", f"- **Memories:** {len(before)}",
           f"- **Note:** {meta.get('note', '-')}", "", "## Observation", "", f"> {obs.text}",
           f"> — {obs.source}, {obs.time}", "", "## Extracted state change", ""]
    d = t.delta
    if d is None:
        out += ["No state change extracted: the observation is stored as a new memory, nothing is repaired.", ""]
    else:
        out += table(["entity", "attribute", "old → new", "change_type", "confidence"],
                     [[d["entity"], d["attribute"], f"{d['old']} → {d['new']}", d["change_type"], f"{d['confidence']:.2f}"]])
        out += ["", f"Roots: {', '.join(t.roots) or 'none'}", "", "## Decisions per candidate", ""]
        out += table(["id", "memory", "hop", "P_dep", "S", "proposed", "final", "gold"],
                     [[m, short(before[m].text), c["hop"], f"{c['p_dep']:.2f}", f"{c['s_evidence']:.2f}",
                       c["proposed"], t.final_actions.get(m, "KEEP"), gold_actions.get(m, "-")]
                      for m, c in t.candidates.items()])
        never = [m for m in before if m not in t.candidates]
        missed = [f"{m} (gold {gold_actions[m]})" for m in never if gold_actions.get(m, "KEEP") != "KEEP"]
        out += ["", f"{len(never)} of {len(before)} memories were never candidates → KEEP by constraint."
                + (f" Gold expected a change for: {', '.join(missed)}." if missed else "")]
        if gold_actions:
            hits = sum(t.final_actions.get(m, "KEEP") == a for m, a in gold_actions.items())
            out.append(f"Final actions matching gold: **{hits}/{len(gold_actions)}**.")
        out += ["", "## Repair sets", ""]
        rows = []
        for r in t.repair_sets:
            s = r["score"]
            acts = ", ".join(f"{m}:{a}" for m, a in r["actions"].items() if a != "KEEP") or "(none)"
            nums = [f"{s[k]:.3f}" for k in ("benefit", "collateral", "evidence_cost")] + [s["size"], f"{s['total']:.3f}"] \
                if s else ["-"] * 5
            status = "yes" if r["feasible"] else f"no ({r['infeasible_reason']})"
            rows.append([f"**{r['name']}**" if r["chosen"] else r["name"], acts, *nums, status,
                         "✅" if r["chosen"] else ""])
        out += table(["set", "actions", "benefit", "collateral", "evidence cost", "size", "total", "feasible", "chosen"], rows)
        out += ["", f"Chosen: **{t.chosen}**.", ""]
        if all(r["score"] for r in t.repair_sets if r["feasible"]):  # scores absent under the use_counterfactual ablation
            out += sensitivity(t.repair_sets, t.chosen, csmr.cfg)

    obs_id = next(x["id"] for x in t.transitions if x["action"] == "ADD")
    without = agent_view(list(before.values()) + [store.get(obs_id)])  # append-only: originals + stored observation
    with_csmr = agent_view(store.all())
    out += ["", "## Answers: without vs with CSMR", "",
            "Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); "
            "*With* = committed store.", ""]
    out += table(["question", "without CSMR", "with CSMR", "gold"],
                 [[q["q"], answer(q["q"], without, cfg=csmr.cfg), answer(q["q"], with_csmr, cfg=csmr.cfg), q["gold_answer"]]
                  for q in (gold.eval_questions if gold else [])])

    out += ["", "## After-state (changed memories)", ""]
    rows = []
    for x in t.transitions:
        m = store.get(x["id"])
        old = before[x["id"]].text if x["id"] in before else "(new)"
        new = m.text + (f" (only valid if: {m.condition})" if m.status == "conditional" and m.condition else "")
        rows.append([x["id"], x["action"], short(old), short(new), f"{x['from_status'] or '-'} → {x['to_status']}",
                     f"v{x['from_version']} → v{x['to_version']}"])
    out += table(["id", "action", "old text", "new text", "status", "version"], rows)
    st = t.llm_stats
    out += ["", f"*Pipeline LLM calls: {st.get('calls', '-')} (cache hits {st.get('hits', '-')}) · "
            f"wall time: {t.wall_time_s:.1f}s*", ""]
    return sid, "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print → / ✅
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("data/showcase").glob("*.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        sid, md = report(p)
        print(md)
        (OUT_DIR / f"{sid}.md").write_text(md, encoding="utf-8")
        print(f"saved {OUT_DIR / f'{sid}.md'}\n")


if __name__ == "__main__":
    try:
        main()
    except llm.QuotaExhausted as e:
        sys.exit(str(e))
