"""CLI: run the CSMR pipeline on one scenario file and print the trace. Gold labels are never passed to the pipeline."""
import argparse
import json
from pathlib import Path

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline.run import run_pipeline
from csmr.scenario import load_scenario


def summary(t: dict) -> str:
    out = [f"observation: {t['observation']['text']!r}"]
    if t["delta"] is None:
        return "\n".join(out + ["no state change -> store untouched", f"llm: {t['llm_stats']}"])
    d = t["delta"]
    out += [f"delta: {d['entity']}.{d['attribute']}: {d['old']!r} -> {d['new']!r} ({d['change_type']}, conf {d['confidence']})",
            f"roots: {t['roots']}", f"probes: {t['probes']}", "candidates:"]
    out += [f"  {m}: hop={c['hop']} P_dep={c['p_dep']:.2f} S={c['s_evidence']:.2f} proposed={c['proposed']}"
            for m, c in t["candidates"].items()]
    out.append("repair sets:")
    out += [f"  {r['name']}{' *' if r['chosen'] else ''}: {r['actions']} "
            + (f"total={r['score']['total']:.3f}" if r["score"] else "" if r["feasible"] else f"infeasible ({r['infeasible_reason']})")
            for r in t["repair_sets"]]
    out += [f"final: {t['final_actions']}", f"transitions: {t['transitions']}", f"validation: {t['validation']}",
            f"llm: {t['llm_stats']}", f"wall: {t['wall_time_s']:.1f}s"]
    return "\n".join(out)


def before_after(t: dict, store: InMemoryAdapter) -> str:
    """Committed changes: before from the history entry the commit pushed, after from the store."""
    cell = lambda text, status, v: f"{text} ({status}, v{v})".replace("|", "\\|")
    out = ["Before -> After", "| id | action | before (text, status, version) | after (text, status, version) |",
           "|---|---|---|---|"]
    for tr in t["transitions"]:  # already = non-KEEP actions + the ADDed observation memory
        m = store.get(tr["id"])
        before = "-" if tr["action"] == "ADD" else cell(m.history[-1]["text"], m.history[-1]["status"], m.history[-1]["version"])
        out.append(f"| {m.id} | {tr['action']} | {before} | {cell(m.text, m.status, m.version)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario")
    ap.add_argument("--trace", help="write the full trace JSON here")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                    help="override a config value, dotted keys allowed (e.g. ablation.use_evidence=false)")
    args = ap.parse_args()
    sc = load_scenario(args.scenario)
    memories, observation = sc.memories, sc.observation
    del sc  # gold lives on the Scenario; drop it so nothing downstream can reach it
    store = InMemoryAdapter(memories)
    trace = run_pipeline(observation, store, load_config(overrides=args.overrides)).to_dict()
    print(summary(trace))
    if trace["delta"] is not None:
        print(before_after(trace, store))
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace).write_text(json.dumps(trace, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except llm.QuotaExhausted as e:
        raise SystemExit(str(e))
