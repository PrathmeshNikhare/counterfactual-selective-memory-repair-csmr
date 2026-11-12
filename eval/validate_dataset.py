"""Benchmark validator (docs/DATASET_NOTE.md). Reads gold labels, so it is eval-only code (CLAUDE.md rule 2).

  python eval/validate_dataset.py [--seeds data/seeds] [--root data/generated]

Seeds: schema (incl. a non-empty `forbidden` list), exactly one valid gold action per core memory, the labeling conventions, dependency / depends_on ids,
balance over categories and domains (overall and per split), every category in every split, splits.json partitions the families.
Scenarios (<root>/<split>/*.json): the same gold checks, plus the real loader accepts the file, 30-40 memories dated
no later than the observation, 3-5 persistence memories dated after it, and the scenario's `family` sits in that
split in splits.json. Near-duplicates (MiniLM cosine > dataset.near_dup_cosine): texts within one family/scenario,
observations across splits (leakage), and every text against data/dev + data/showcase. Prints count tables per
split x category / domain / action; exits 1 on any error."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from csmr.config import ROOT, load_config
from csmr.models import embed
from csmr.scenario import load_scenario
from csmr.types import Action

CATEGORIES = ("direct_contradiction", "one_hop", "two_hop", "three_hop", "false_similarity", "weak_evidence",
              "conflicting_source", "temporary_change")
DOMAINS = ("vehicle", "job", "residence", "health_habit", "diet", "pet", "relationship", "education", "devices",
           "finances")
SOURCES = {"user", "tool", "third_party", "inferred"}
ROLES = {"root", "dependent", "hard_negative"}
ACTIONS = {a.value for a in Action}
MEM_KEYS = {"id", "text", "entity", "attribute", "time", "source"}
SEED_MEM_KEYS = {"id", "text", "entity", "attribute", "source", "role", "gold_action"}
SEED_KEYS = {"family_id", "category", "domain", "pattern", "observation", "core_memories", "dependencies",
             "hard_negative_guidance", "hard_negative_examples", "forbidden", "persistence_guidance", "persistence_examples",
             "eval_questions"}
SCENARIO_KEYS = {"id", "category", "domain", "family", "memories", "observation", "gold"}


def check_gold(where: str, cat: str, dom: str, mem_ids: list[str], actions: dict, deps: list,
               questions: list[dict]) -> list[str]:
    """Checks shared by seed families and scenarios (DATASET_NOTE labeling conventions)."""
    e = [f"{where}: unknown category {cat!r}"] * (cat not in CATEGORIES)
    e += [f"{where}: unknown domain {dom!r}"] * (dom not in DOMAINS)
    e += [f"{where}: duplicate memory ids"] * (len(set(mem_ids)) != len(mem_ids))
    ids = set(mem_ids)
    if set(actions) != ids:
        e.append(f"{where}: gold actions do not cover exactly the memories: {sorted(set(actions) ^ ids)}")
    e += [f"{where}: {m} has invalid action {a!r}" for m, a in actions.items() if a not in ACTIONS]
    e += [f"{where}: dependency {p} is not a pair of known ids" for p in deps if len(p) != 2 or not set(p) <= ids]
    dependents = {p[1] for p in deps}
    e += [f"{where}: root {r} is KEEP" for r in sorted({p[0] for p in deps} - dependents) if actions.get(r) == "KEEP"]
    e += [f"{where}: dependent {d} is REVOKE (roots only)" for d in sorted(dependents) if actions.get(d) == "REVOKE"]
    for q in questions:
        if not (q.get("q") and q.get("gold_answer")):
            e.append(f"{where}: eval question without q or gold_answer: {q}")
        if not q.get("depends_on") or not set(q["depends_on"]) <= ids:
            e.append(f"{where}: depends_on {q.get('depends_on')} of {q.get('q')!r} is missing or names unknown ids")
    used = set(actions.values())
    if cat == "weak_evidence" and used & {"REVISE", "REVOKE"}:
        e.append(f"{where}: weak_evidence must not REVISE/REVOKE")
    if cat == "temporary_change" and used - {"KEEP", "CONDITIONAL"}:
        e.append(f"{where}: temporary_change uses only KEEP/CONDITIONAL")
    if cat != "temporary_change" and "CONDITIONAL" in used:
        e.append(f"{where}: CONDITIONAL only in temporary_change")
    return e


def _texts(memories: list[dict], obs: dict, extra: list[str] = ()) -> list[str]:
    """Stored-style texts (memories, observation, examples). Eval questions are kept apart: they echo their memory."""
    return [m["text"] for m in memories] + [obs["text"]] + list(extra)


def _balanced(where: str, counts: Counter, keys) -> list[str]:
    n = [counts[k] for k in keys]
    return [f"{where} unbalanced: {dict(counts)}"] * (max(n) - min(n) > 1)


def check_seeds(seed_dir: Path) -> tuple[list[str], list[dict], dict[str, str]]:
    """Errors, one item per family (for near-dups and stats), family -> split."""
    fams = json.loads((seed_dir / "seeds.json").read_text(encoding="utf-8"))["families"]
    splits = json.loads((seed_dir / "splits.json").read_text(encoding="utf-8"))
    split_of = {f: s for s, fs in splits.items() for f in fs}
    ids = [f.get("family_id") for f in fams]
    e = [f"seeds: duplicate family ids"] * (len(set(ids)) != len(ids))
    if sum(map(len, splits.values())) != len(split_of) or set(split_of) != set(ids):
        e.append(f"splits.json does not partition the families: {sorted(set(split_of) ^ set(ids))}")
    items = []
    for f in fams:
        w = f"seed {f.get('family_id')}"
        if SEED_KEYS - f.keys():
            e.append(f"{w}: missing keys {sorted(SEED_KEYS - f.keys())}")
            continue
        mems = f["core_memories"]
        for m in mems:
            e += [f"{w}: memory {m.get('id')} missing keys {sorted(SEED_MEM_KEYS - m.keys())}"] * bool(SEED_MEM_KEYS - m.keys())
            e += [f"{w}: memory {m.get('id')} has bad source/role"] * (m.get("source") not in SOURCES
                                                                         or m.get("role") not in ROLES)
        e += [f"{w}: forbidden must be a non-empty list of strings"] * (
            not f["forbidden"] or not all(isinstance(x, str) and x for x in f["forbidden"]))
        e += [f"{w}: observation needs text and source"] * (not {"text", "source"} <= f["observation"].keys())
        e += check_gold(w, f["category"], f["domain"], [m["id"] for m in mems],
                        {m["id"]: m.get("gold_action") for m in mems}, f["dependencies"], f["eval_questions"])
        items.append({"name": w, "split": split_of.get(f["family_id"], "?"), "category": f["category"],
                      "domain": f["domain"], "obs": f["observation"]["text"],
                      "actions": {m["id"]: m.get("gold_action") for m in mems},
                      "texts": _texts(mems, f["observation"], f["hard_negative_examples"] + f["persistence_examples"]),
                      "questions": [q["q"] for q in f["eval_questions"]]})
    e += _balanced("seeds: categories", Counter(i["category"] for i in items), CATEGORIES)
    e += _balanced("seeds: domains", Counter(i["domain"] for i in items), DOMAINS)
    for s in splits:
        e += [f"split {s}: missing categories {sorted(set(CATEGORIES) - {i['category'] for i in items if i['split'] == s})}"] * bool(
            set(CATEGORIES) - {i["category"] for i in items if i["split"] == s})
        e += _balanced(f"split {s}: categories", Counter(i["category"] for i in items if i["split"] == s), CATEGORIES)
        e += _balanced(f"split {s}: domains", Counter(i["domain"] for i in items if i["split"] == s), DOMAINS)
    return e, items, split_of


def _time(s) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def check_scenario(path: Path, split: str, split_of: dict[str, str], lim: dict) -> tuple[list[str], dict | None]:
    w = f"{split}/{path.name}"
    d = json.loads(path.read_text(encoding="utf-8"))
    if SCENARIO_KEYS - d.keys():
        return [f"{w}: missing keys {sorted(SCENARIO_KEYS - d.keys())}"], None
    e = []
    try:
        load_scenario(path)
    except Exception as ex:  # the real loader is the schema authority
        e.append(f"{w}: load_scenario rejects it: {ex!r}")
    obs_t = _time(d["observation"].get("time"))
    e += [f"{w}: observation time missing or not ISO"] * (obs_t is None)
    mems, g = d["memories"], d["gold"]
    for m in mems:
        if MEM_KEYS - m.keys() or m.get("source") not in SOURCES:
            e.append(f"{w}: memory {m.get('id')} missing keys {sorted(MEM_KEYS - m.keys())} or bad source")
        elif obs_t and (_time(m["time"]) is None or _time(m["time"]) > obs_t):
            e.append(f"{w}: memory {m['id']} time {m['time']} is not ISO or is after the observation")
    if not lim["memories_min"] <= len(mems) <= lim["memories_max"]:
        e.append(f"{w}: {len(mems)} memories, need {lim['memories_min']}-{lim['memories_max']}")
    e += check_gold(w, d["category"], d["domain"], [m.get("id") for m in mems], g.get("actions", {}),
                    g.get("dependencies", []), g.get("eval_questions", []))
    pers = g.get("persistence_memories", [])
    if not lim["persistence_min"] <= len(pers) <= lim["persistence_max"]:
        e.append(f"{w}: {len(pers)} persistence_memories, need {lim['persistence_min']}-{lim['persistence_max']}")
    for p in pers:
        t = _time(p.get("time"))
        if MEM_KEYS - p.keys() or not str(p.get("id", "")).startswith("P") or t is None or (obs_t and t <= obs_t):
            e.append(f"{w}: persistence memory {p.get('id')} needs all fields, a P id and a time after the observation")
    if split_of.get(d["family"]) != split:
        e.append(f"{w}: family {d['family']!r} is in split {split_of.get(d['family'])!r}, not {split!r}")
    item = {"name": w, "split": split, "category": d["category"], "domain": d["domain"],
            "obs": d["observation"]["text"], "actions": g.get("actions", {}),
            "texts": _texts(mems + pers, d["observation"]), "questions": [q["q"] for q in g.get("eval_questions", [])]}
    return e, item


def reference_texts() -> list[str]:
    """Every text in data/dev and data/showcase: the benchmark must not reuse their wording."""
    out = []
    for p in sorted([*(ROOT / "data/dev").glob("*.json"), *(ROOT / "data/showcase").glob("*.json")]):
        d = json.loads(p.read_text(encoding="utf-8"))
        g = d.get("gold") or {}
        out += _texts(d["memories"] + g.get("persistence_memories", []), d["observation"])
        out += [q["q"] for q in g.get("eval_questions", [])]
    return list(dict.fromkeys(out))


def _pairs(a: np.ndarray, b: np.ndarray, thr: float, upper: bool = False):
    s = a @ b.T
    if upper:
        s = np.triu(s, 1)
    return [(i, j, s[i, j]) for i, j in zip(*np.where(s > thr))]


def near_dups(items: list[dict], ref: list[str], thr: float) -> list[str]:
    e = []
    r = embed(ref)
    for it in items:
        t = it["texts"] + it["questions"]
        v = embed(t)
        n = len(it["texts"])  # within an item: stored texts only; against dev/showcase: questions too
        e += [f"{it['name']}: near-duplicate texts ({s:.2f}): {t[i]!r} / {t[j]!r}"
              for i, j, s in _pairs(v[:n], v[:n], thr, True)]
        e += [f"{it['name']}: too close to dev/showcase ({s:.2f}): {t[i]!r} / {ref[j]!r}" for i, j, s in _pairs(v, r, thr)]
    v = embed([it["obs"] for it in items])
    e += [f"observations across splits ({s:.2f}): {items[i]['name']} / {items[j]['name']}"
          for i, j, s in _pairs(v, v, thr, True) if items[i]["split"] != items[j]["split"]]
    return e


def stats(items: list[dict]) -> str:
    df = pd.DataFrame(items)
    acts = pd.DataFrame([{"split": it["split"], "action": a} for it in items for a in it["actions"].values()])
    return "\n\n".join(pd.crosstab(t[k], t["split"], margins=True, margins_name="all").to_string()
                       for t, k in ((df, "category"), (df, "domain"), (acts, "action")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="data/seeds")
    ap.add_argument("--root", default="data/generated")
    args = ap.parse_args(argv)
    lim = load_config()["dataset"]
    errs, seed_items, split_of = check_seeds(ROOT / args.seeds)
    scen_items = []
    for path in sorted((ROOT / args.root).glob("*/*.json")):
        e, item = check_scenario(path, path.parent.name, split_of, lim)
        errs += e
        scen_items += [item] if item else []
    ref = reference_texts()
    for label, items in (("seed families", seed_items), ("scenarios", scen_items)):
        if items:
            errs += near_dups(items, ref, lim["near_dup_cosine"])
            print(f"== {label}: {len(items)}\n{stats(items)}\n")
    print("\n".join(errs) or "OK: no errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
