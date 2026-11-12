"""Fit the Phase C classifiers on train and tune tau_d, tau_e, lambda_c, lambda_s, lambda_e on val (TASKS E3;
Paper IV-C, Section V). Training code: it reads gold labels, which only eval/ and fit code may do (CLAUDE.md rule 2).

  python eval/fit.py train [--shard i/n]   Phases A-C on every train scenario -> one feature-row file each
                                           (results/fit/<tag>/train/<id>.json, resumable); then, unsharded, fit both
                                           logistic regressions -> models/dependency.json, models/evidence.json
  python eval/fit.py val [--shard i/n]     CSMR once on val with the fitted models, through eval/run_all.py
                                           (raw format, sharding, resume) -> results/raw/<tag>/val/csmr/<id>.json
  python eval/fit.py tune                  grid search on the saved val traces -> config/tuned.yaml,
                                           results/fit/<tag>/grid.csv (every grid point)

Training rows (labels from gold):
- dependency: every non-root candidate with features; y = it is a dependent in gold.dependencies. Pipeline roots
  (P_dep = 1 by construction) are left out, as in the dependency metric (Spec 14), and so are gold roots that Phase A
  missed (they are the memories most affected, not "independent"). Features are the trace's: for a candidate with
  several roots, those of the root with the highest P_dep under the weights active at extraction.
- evidence: every candidate the change affects: gold action not KEEP, or a gold dependent (a dependent kept because
  the evidence is weak, e.g. "maybe"); y = gold is REVISE or REVOKE, i.e. the evidence was enough to modify it
  (VERIFY = not enough). Other gold-KEEP candidates are unaffected whatever the evidence, and gold CONDITIONAL is
  decided by the temporary flag, not evidence strength: both carry no evidence label and are left out.
--config-overlay PATH (all steps): yaml loaded on top of config.yaml and included in the config hash. tune rebuilds
the val run's config from the overlay and --set items recorded in its raw files, not from its own flags.

Tuning (offline where possible): tau_d / tau_e change the proposed actions and hence R0..R4 and feasibility; lambdas
change only the objective. For every grid point the repair sets are rebuilt from the saved P_dep / S
(pipeline.actions.propose_action, pipeline.repair_sets.generate_repair_sets), each feasible set's Benefit is taken from
the saved trace when the val run scored a set with the same actions, and the objective / argmax are the pipeline's own
(counterfactual.objective, counterfactual.argmax). A set the val run never scored gets its Benefit online with the
saved probes (LLM calls, disk-cached), unless --offline-only, which leaves that grid point unscored. Approximation:
probes stay those of the val run, although tau_d (eligibility) and tau_e (capping priority, via the proposed actions)
also decide which candidates get a probe. Refuses if config.yaml / models changed since the val run (config hash).
Selection: maximise pooled repair F1 - over-invalidation over val; ties -> first point in grid order."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from dotenv import load_dotenv

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import ROOT, load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline import dependency, evidence
from csmr.pipeline.actions import propose_action
from csmr.pipeline.overlay import observation_memory
from csmr.pipeline.repair_sets import generate_repair_sets
from csmr.pipeline.run import phase_a, phase_bc
from csmr.scenario import GoldLabels, Scenario, load_scenario
from csmr.types import MODIFYING_ACTIONS, Action, CandidateInfo, Observation, RepairSet, StateChange
from eval import aggregate, run_all
from eval.metrics import Record, repair

TUNED = ROOT / "config" / "tuned.yaml"
GRID_KEYS = ("tau_d", "tau_e", "lambda_c", "lambda_s", "lambda_e")  # each searched over cfg["fit"]["<key>_grid"]


def _fit_dir(tag: str) -> Path:
    return run_all.RESULTS / "fit" / tag


def _full_method(cfg: dict) -> bool:
    return all(cfg["ablation"].values())


# ---------------------------------------------------------------- (1) train: features + logistic regressions

def label_rows(cands: dict[str, CandidateInfo], gold: GoldLabels) -> dict[str, list[dict]]:
    """Dependency and evidence training rows for one scenario (labelling: module docstring)."""
    gold_dep = {d for _, d in gold.dependencies}
    gold_roots = {r for r, _ in gold.dependencies} - gold_dep
    acts = {m: Action(a) for m, a in gold.actions.items()}  # strict: an unknown label raises
    dep = [{"memory": m, "features": i.dep_features, "label": int(m in gold_dep)}
           for m, i in cands.items() if not i.is_root and m not in gold_roots and i.dep_features]
    ev = [{"memory": m, "features": i.ev_features, "label": int(acts[m] in MODIFYING_ACTIONS)}
          for m, i in cands.items()
          if (acts[m] != Action.KEEP or m in gold_dep) and acts[m] != Action.CONDITIONAL and i.ev_features]
    return {"dependency": dep, "evidence": ev}


def extract_rows(sc: Scenario, cfg: dict) -> dict[str, list[dict]]:
    """Phases A-C on a fresh store (the pipeline sees observation + store only), then label with gold."""
    store = InMemoryAdapter(copy.deepcopy(sc.memories))
    delta, _, roots = phase_a(sc.observation, store, cfg)
    if delta is None:
        return {"dependency": [], "evidence": []}
    return label_rows(phase_bc(delta, sc.observation, roots, store, cfg), sc.gold)


def train(tag: str, shard_spec: str, cfg: dict, force: bool = False, overlay: str | None = None) -> dict | None:
    """Extract missing feature rows for this shard; fit when unsharded. Returns the fit summary (None if sharded).
    cfg must already include `overlay`; it is passed for the hash."""
    fcfg = copy.deepcopy(cfg)
    fcfg["ablation"] = {k: True for k in fcfg["ablation"]}  # features need every stage on
    out_dir = _fit_dir(tag) / "train"
    meta = {"config_hash": run_all.config_hash(fcfg, overlay), "git_commit": run_all.git_commit(),
            "config_overlay": overlay}
    for path in run_all.shard(run_all.scenario_paths("train"), shard_spec):
        sc = load_scenario(path)
        out = out_dir / f"{sc.id}.json"
        if out.exists():
            continue
        try:
            rows = extract_rows(sc, fcfg)
        except llm.QuotaExhausted as e:
            raise SystemExit(f"{e}. Stopped at train/{sc.id}; rerun the same command to resume.")
        except Exception as e:  # one bad scenario must not stop the run
            print(f"ERROR train/{sc.id}: {e!r}", file=sys.stderr)
            continue
        run_all._write(out, {**meta, "scenario": sc.id, **rows})
        print(f"train/{sc.id}: {len(rows['dependency'])} dependency rows, {len(rows['evidence'])} evidence rows")
    if shard_spec != "1/1":
        print("sharded: features only; run `train` without --shard to fit")
        return None
    return fit_models(out_dir, force)


def fit_models(rows_dir: Path, force: bool = False) -> dict:
    """Fit both classifiers on every row file (refuses mixed config hashes unless force); writes models/*.json
    (via fit_dependency / fit_evidence) and <rows_dir>/../train_summary.json."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rows_dir.glob("*.json"))]
    if not recs:
        raise SystemExit(f"no feature rows under {rows_dir}")
    if len(hashes := {r["config_hash"] for r in recs}) > 1 and not force:
        raise SystemExit(f"feature rows have different config hashes {sorted(hashes)}; re-extract or pass --force")
    from sklearn.metrics import roc_auc_score

    summary = {"scenarios": len(recs), "expected": len(run_all.scenario_paths("train"))}
    for name, mod, fit, prob in (("dependency", dependency, dependency.fit_dependency, dependency.dependency_prob),
                                 ("evidence", evidence, evidence.fit_evidence, evidence.evidence_score)):
        X = [row["features"] for r in recs for row in r[name]]
        y = [row["label"] for r in recs for row in r[name]]
        if len(set(y)) < 2:
            raise SystemExit(f"{name}: need both labels to fit, got {len(y)} rows with labels {sorted(set(y))}")
        fit(X, y)
        summary[name] = {"rows": len(y), "positives": sum(y), "train_auc": roc_auc_score(y, [prob(x) for x in X]),
                         "weights": json.loads(mod._MODEL_FILE.read_text())}
    (rows_dir.parent / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("dependency", "evidence")}
                     | {n: {k: summary[n][k] for k in ("rows", "positives", "train_auc")}
                        for n in ("dependency", "evidence")}, indent=2))
    return summary


# ---------------------------------------------------------------- (2a) val: one CSMR run, run_all raw format

def val(tag: str, shard_spec: str, cfg: dict, overrides: list[str], allow_handset: bool = False,
        overlay: str | None = None) -> int:
    """CSMR on val with the fitted classifiers -> results/raw/<tag>/val/csmr/ (run_all; resumable, shardable)."""
    if not _full_method(cfg):
        raise SystemExit(f"tuning needs the full method; ablation flags are {cfg['ablation']}")
    missing = [str(p) for p in (dependency._MODEL_FILE, evidence._MODEL_FILE) if not p.exists()]
    if missing and not allow_handset:
        raise SystemExit(f"fitted models missing {missing}; run `fit.py train` first or pass --allow-handset")
    return run_all.run("val", ["csmr"], tag, shard_spec, cfg, overrides, overlay)


# ---------------------------------------------------------------- (2b) tune: grid search on saved traces

@dataclass
class Saved:
    """One val trace, reduced to what re-scoring needs."""
    scenario: Scenario
    delta: StateChange | None
    obs: Observation
    cands: dict[str, CandidateInfo]
    probes: list[dict]
    benefit: dict[tuple, float]  # RepairSet.key() -> Benefit the val run measured (filled sets added)
    cache: dict = field(default_factory=dict)  # counterfactual answer cache for online fills


def load_saved(rec: dict, path: Path) -> Saved:
    t = rec["trace"]
    cands = {m: CandidateInfo(**{**c, "proposed": Action(c["proposed"])}) for m, c in t["candidates"].items()}
    benefit = {RepairSet(r["name"], {k: Action(v) for k, v in r["actions"].items()}).key(): r["score"]["benefit"]
               for r in t["repair_sets"] if "benefit" in r.get("score", {})}
    return Saved(load_scenario(path), StateChange(**t["delta"]) if t["delta"] else None,
                 Observation(**t["observation"]), cands, t["probes"], benefit)


def fill_benefit(s: Saved, rs: RepairSet, cfg: dict) -> float:
    """Benefit of a set the val run never scored: the shared agent answers the saved probes over its view (LLM)."""
    store = InMemoryAdapter(copy.deepcopy(s.scenario.memories))
    new_mem = observation_memory(s.delta, s.obs, store)
    revised, conditions = cf.prepare_texts([rs], store, s.delta, s.obs, cfg)
    return cf.score(rs, s.probes, store, revised, conditions, s.cands, s.delta, s.obs, cfg, s.cache, new_mem)["benefit"]


def feasible_sets(s: Saved, cfg: dict, online: bool, stats: dict) -> list[RepairSet] | None:
    """Feasible R0..R4 under cfg's tau_d / tau_e, each with score["benefit"]; [] when there is no state change;
    None when a Benefit is unknown and online is False."""
    if s.delta is None:
        return []
    proposed = {m: propose_action(m, i, cfg, s.delta.change_type) for m, i in s.cands.items()}
    out = []
    for rs in generate_repair_sets(s.cands, proposed, cfg):
        if not rs.feasible:
            continue
        if rs.key() not in s.benefit:
            if not online:
                return None
            s.benefit[rs.key()] = fill_benefit(s, rs, cfg)
            stats["filled"] += 1
        rs.score = {"benefit": s.benefit[rs.key()]}
        out.append(rs)
    return out


def choose(s: Saved, sets: list[RepairSet], cfg: dict) -> dict[str, str]:
    """Final actions under cfg's lambdas: the pipeline's objective + argmax over the feasible sets."""
    if not sets:
        return {m.id: Action.KEEP.value for m in s.scenario.memories}
    for rs in sets:
        rs.score = cf.objective(rs.score["benefit"], rs, s.cands, cfg)
    best = cf.argmax(sets, cfg)
    return {m.id: best.actions.get(m.id, Action.KEEP).value for m in s.scenario.memories}


def point_metrics(saved: list[Saved], finals: list[dict[str, str] | None]) -> dict:
    """Pooled repair F1, over-invalidation and the objective F1 - OI (NaN unless every scenario was scored)."""
    counts = [repair(Record(s.scenario, f)) for s, f in zip(saved, finals) if f is not None]
    tot = lambda k, i: sum(c[k][i] for c in counts)
    tp, changed, gold_changed = tot("repair_recall", 0), tot("repair_precision", 1), tot("repair_recall", 1)
    oi_n, oi_d = tot("over_invalidation", 0), tot("over_invalidation", 1)
    ratio = lambda a, b: a / b if b else float("nan")
    f1 = 2 * tp / (changed + gold_changed) if changed + gold_changed else 0.0  # no changes anywhere: F1 = 0
    oi = oi_n / oi_d if oi_d else 0.0
    full = len(counts) == len(saved)
    return {"n_scored": len(counts), "repair_precision": ratio(tp, changed), "repair_recall": ratio(tp, gold_changed),
            "repair_f1": f1, "over_invalidation": oi, "objective": f1 - oi if full else float("nan")}


def tune(tag: str, cfg: dict, offline_only: bool = False, out: Path | None = None) -> dict:
    """Grid search over cfg["fit"] grids on the csmr val traces of `tag`; writes the tuned yaml and grid.csv.
    Re-scoring uses the val run's own config (from the raw files) with the five tuned keys replaced."""
    recs = [r for r in aggregate.load(tag, "val") if r["method"] == "csmr"]
    if not recs:
        raise SystemExit(f"no csmr val results under tag {tag!r}; run `fit.py val` first")
    overlay = recs[0].get("config_overlay")  # aggregate.load refused mixed config hashes
    base = load_config(overrides=recs[0]["overrides"], overlay=overlay)
    if run_all.config_hash(base, overlay) != recs[0]["config_hash"]:
        raise SystemExit("config.yaml, the config overlay or models/*.json changed since the val run; "
                         "rerun `fit.py val` with a new --tag")
    if not _full_method(base):
        raise SystemExit(f"val run used ablation flags {base['ablation']}; tuning needs the full method")
    paths = {load_scenario(p).id: p for p in run_all.scenario_paths("val")}
    if len(recs) < len(paths):
        print(f"WARNING: tuning on {len(recs)} of {len(paths)} val scenarios (the rest have no csmr result; "
              f"see results/raw/{tag}/errors.jsonl)", file=sys.stderr)
    saved = [load_saved(r, paths[r["scenario"]]) for r in recs]
    grids = {k: cfg["fit"][f"{k}_grid"] for k in GRID_KEYS}
    stats, rows = {"filled": 0}, []
    try:
        for td, te in product(grids["tau_d"], grids["tau_e"]):
            sets = [feasible_sets(s, {**base, "tau_d": td, "tau_e": te}, not offline_only, stats) for s in saved]
            for lc, ls, le in product(grids["lambda_c"], grids["lambda_s"], grids["lambda_e"]):
                point = {"tau_d": td, "tau_e": te, "lambda_c": lc, "lambda_s": ls, "lambda_e": le}
                pcfg = {**base, **point}
                finals = [None if st is None else choose(s, st, pcfg) for s, st in zip(saved, sets)]
                rows.append({**point, **point_metrics(saved, finals)})
    except llm.QuotaExhausted as e:
        raise SystemExit(f"{e}. Rerun `fit.py tune`: answers already computed are in the LLM disk cache.")
    grid = pd.DataFrame(rows)
    grid.insert(len(GRID_KEYS), "n_scenarios", len(saved))
    scored = grid["objective"].notna()
    if not scored.any():
        raise SystemExit("no grid point could be scored on every val scenario; rerun without --offline-only")
    best = grid.loc[grid["objective"][scored].idxmax()]  # idxmax: first maximum = first in grid order
    grid["chosen"] = grid.index == best.name
    fit_dir = _fit_dir(tag)
    fit_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(fit_dir / "grid.csv", index=False)
    tuned = {k: float(best[k]) for k in GRID_KEYS}
    out = out or TUNED
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Tuned on val by eval/fit.py tune: tag {tag!r}, {len(saved)} of {len(paths)} scenarios, raw config hash "
              f"{recs[0]['config_hash']}, git {recs[0]['git_commit']}.\n"
              f"# repair F1 {best['repair_f1']:.4f} - over-invalidation {best['over_invalidation']:.4f} = "
              f"{best['objective']:.4f}. Full grid: {fit_dir / 'grid.csv'}.\n"
              f"# Use: eval/run_all.py --config-overlay {out.as_posix()}\n")
    out.write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    print(f"{len(grid)} grid points, {int(scored.sum())} scored, {stats['filled']} Benefits computed online; "
          f"chosen {tuned} objective {best['objective']:.4f} -> {out}")
    return {"tuned": tuned, "objective": float(best["objective"]), "filled": stats["filled"], "grid": grid}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "val", "tune"])
    ap.add_argument("--tag", default="fit", help="results/fit/<tag>/ and results/raw/<tag>/val/")
    ap.add_argument("--shard", default="1/1", help="i/n, 1-based (train, val)")
    ap.add_argument("--force", action="store_true", help="train: fit rows with different config hashes")
    ap.add_argument("--allow-handset", action="store_true", help="val: run without fitted models/*.json")
    ap.add_argument("--offline-only", action="store_true", help="tune: never call the LLM; unscorable points -> NaN")
    ap.add_argument("--out", type=Path, help=f"tune: output yaml (default {TUNED.relative_to(ROOT)})")
    ap.add_argument("--env-file", help="dotenv file with LLM_API_KEY / LLM_BASE_URL (overrides .env)")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    ap.add_argument("--config-overlay", metavar="PATH",
                    help="yaml loaded on top of config.yaml (e.g. config/tuned.yaml); --set still wins")
    args = ap.parse_args(argv)
    if args.env_file:
        if not Path(args.env_file).is_file():
            ap.error(f"--env-file not found: {args.env_file}")
        load_dotenv(args.env_file, override=True)
    cfg = load_config(overrides=args.overrides, overlay=args.config_overlay)
    if args.step == "train":
        train(args.tag, args.shard, cfg, args.force, args.config_overlay)
    elif args.step == "val":
        n = val(args.tag, args.shard, cfg, args.overrides, args.allow_handset, args.config_overlay)
        print(f"done: {n} new results")
    else:
        tune(args.tag, cfg, args.offline_only, args.out)


if __name__ == "__main__":
    main()
