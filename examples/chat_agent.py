"""Terminal demo: a line ending in '?' is a question to the agent; any other line is an observation (CSMR repairs memory).

    python examples/chat_agent.py [data/dev/car_sale_distractors.json] [--set lambda_s=0.01 ...]

Type e.g. "I sold my car", then "how do I get to work?". Empty line / Ctrl-D quits.
Gold labels in the scenario file are never read: only its memories seed the store.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csmr import llm  # noqa: E402
from csmr.adapters.memory import InMemoryAdapter  # noqa: E402
from csmr.api import CSMR  # noqa: E402
from csmr.config import load_config  # noqa: E402
from csmr.pipeline.agent import answer  # noqa: E402
from csmr.scenario import load_scenario  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", default="data/dev/car_sale_distractors.json")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                    help="override a config value, dotted keys allowed (e.g. ablation.use_evidence=false)")
    args = ap.parse_args()
    sc = load_scenario(args.scenario)
    memories = sc.memories
    del sc  # gold lives on the Scenario; drop it so nothing downstream can reach it
    csmr = CSMR(InMemoryAdapter(memories), load_config(overrides=args.overrides))
    print(f"{len(memories)} memories loaded. State a change, or ask a question ('?' at the end).")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line.endswith("?"):
            print("agent>", answer(line, csmr.retrieve_memories(line), cfg=csmr.cfg))
            continue
        t = csmr.observe(line)
        for x in t.transitions:
            if x["action"] == "ADD":
                print(f"stored {x['id']}: {csmr.store.get(x['id']).text}")
        d = t.delta
        print(f"delta: {d['entity']}.{d['attribute']}: {d['old']!r} -> {d['new']!r} ({d['change_type']})" if d else "delta: no change")
        if d:
            print("chosen set:", t.chosen, "| repairs:", {m: a for m, a in t.final_actions.items() if a != "KEEP"} or "none")
        if t.llm_stats["fallbacks"]:
            print("fallbacks:", t.llm_stats["fallbacks"])
        for m in csmr.pending_verifications():
            print(f"  pending verification: {m.id}: {m.text}")


if __name__ == "__main__":
    try:
        main()
    except llm.QuotaExhausted as e:
        sys.exit(str(e))
