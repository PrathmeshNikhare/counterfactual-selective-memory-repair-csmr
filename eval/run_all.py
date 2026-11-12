"""Run CSMR and the baselines on one split and save one raw JSON per (method, scenario) (Spec 14; TASKS E4).

Output: results/raw/<tag>/<split>/<method>/<scenario id>.json. Existing files are skipped, so a rerun resumes.
Gold is read only here (via eval.metrics), after the method has run; methods get the observation and a store only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from baselines import BASELINES
from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import ROOT, load_config
from csmr.pipeline.run import run_pipeline
from csmr.scenario import load_scenario
from eval.metrics import Record, scenario_metrics

SPLITS = {"train": "data/generated/train", "val": "data/generated/val", "test": "data/generated/test",
          "dev": "data/dev", "showcase": "data/showcase"}
METHODS = {"csmr": run_pipeline, **BASELINES}
RESULTS = ROOT / "results"


def scenario_paths(split: str) -> list[Path]:
    paths = sorted((ROOT / SPLITS[split]).glob("*.json"))
    if not paths:
        raise SystemExit(f"no scenarios in {SPLITS[split]}")
    return paths


def shard(paths: list[Path], spec: str) -> list[Path]:
    """--shard i/n (1-based): every n-th scenario starting at the i-th, so shards 1..n partition the split."""
    i, n = map(int, spec.split("/"))
    if not 1 <= i <= n:
        raise ValueError(f"--shard must be i/n with 1 <= i <= n, got {spec!r}")
    return paths[i - 1::n]


def config_hash(cfg: dict) -> str:
    """Effective config plus any fitted weights (models/*.json are gitignored, so the commit does not cover them)."""
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode())
    for p in sorted((ROOT / "models").glob("*.json")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_one(method: str, path: Path, cfg: dict) -> dict:
    """Run one method on a fresh store, then score it. Only observation + store reach the method."""
    sc = load_scenario(path)
    store = InMemoryAdapter(sc.memories)
    trace = METHODS[method](sc.observation, store, cfg)
    llm.reset_stats()  # from here on: the eval judge's calls, kept apart from the method's
    counts = scenario_metrics(Record(sc, trace.final_actions, trace, store), cfg)
    return {"method": method, "scenario": sc.id, "category": sc.meta.get("category"),
            "final_actions": trace.final_actions, "trace": trace.to_dict(), "metrics": counts,
            "llm_stats": trace.llm_stats, "eval_llm_stats": llm.stats()}


def _write(path: Path, obj: dict) -> None:
    """Atomic, so an interrupted run never leaves a partial file that resume would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run(split: str, methods: list[str], tag: str, shard_spec: str, cfg: dict, overrides: list[str]) -> int:
    """Returns the number of new result files. Raises SystemExit (with a message) on QuotaExhausted."""
    out_dir, errors = RESULTS / "raw" / tag / split, RESULTS / "raw" / tag / "errors.jsonl"
    meta = {"split": split, "tag": tag, "config_hash": config_hash(cfg), "git_commit": git_commit(),
            "overrides": overrides}
    done = 0
    for path in shard(scenario_paths(split), shard_spec):  # scenario-outer: a stopped shard has all methods per scenario
        sid = load_scenario(path).id
        for method in methods:
            out = out_dir / method / f"{sid}.json"
            if out.exists():
                continue
            try:
                rec = run_one(method, path, cfg)
            except llm.QuotaExhausted as e:
                raise SystemExit(f"{e}. Shard {shard_spec} stopped at {method}/{sid} after {done} new results; "
                                 "rerun the same command to resume.")
            except Exception as e:  # one bad scenario must not stop the run
                errors.parent.mkdir(parents=True, exist_ok=True)
                with errors.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**meta, "method": method, "scenario": sid, "error": repr(e),
                                        "traceback": traceback.format_exc()}) + "\n")
                print(f"ERROR {method}/{sid}: {e!r} (logged to {errors})", file=sys.stderr)
                continue
            _write(out, {**meta, **rec})
            done += 1
            print(f"{method}/{sid}: {rec['final_actions']}")
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--methods", default="all", help=f'comma list or "all" = {",".join(METHODS)}')
    ap.add_argument("--shard", default="1/1", help="i/n, 1-based")
    ap.add_argument("--env-file", help="dotenv file with LLM_API_KEY / LLM_BASE_URL (overrides .env)")
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    ap.add_argument("--tag", default="main", help='label for the output folder, e.g. "main", "abl_nodep"')
    args = ap.parse_args(argv)
    methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
    if bad := [m for m in methods if m not in METHODS]:
        ap.error(f"unknown methods {bad}; choose from {list(METHODS)}")
    if args.env_file:
        if not Path(args.env_file).is_file():
            ap.error(f"--env-file not found: {args.env_file}")
        load_dotenv(args.env_file, override=True)  # llm.py's later load_dotenv(.env) does not override
    n = run(args.split, methods, args.tag, args.shard, load_config(overrides=args.overrides), args.overrides)
    print(f"done: {n} new results")


if __name__ == "__main__":
    main()
