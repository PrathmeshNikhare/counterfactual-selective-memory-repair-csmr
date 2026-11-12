"""Merge raw per-scenario results (eval/run_all.py) into results/<tag>/<split>/: main.csv, per_category.csv, persistence.csv,
cost.csv, summary.md. Values are pooled with 95% scenario-level bootstrap CIs (eval.metrics.aggregate, Spec 14)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from csmr.config import load_config
from eval.metrics import aggregate
from eval.run_all import RESULTS

MAIN = ["action_accuracy", "repair_precision", "repair_recall", "over_invalidation", "under_repair",
        "dependent_precision", "dependent_recall", "dependent_f1", "candidate_recall", "qa_accuracy"]
PERSISTENCE = ["qa_accuracy", "persistence", "persistence_fallback"]
COST = ["llm_calls", "wall_time_s"]
SUMMARY = ["action_accuracy", "repair_precision", "repair_recall", "over_invalidation", "qa_accuracy", "persistence"]


def load(tag: str, split: str, force: bool = False) -> list[dict]:
    """All raw records; refuses (SystemExit) to mix config hashes or git commits unless force."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS / "raw" / tag / split).glob("*/*.json"))]
    if not recs:
        raise SystemExit(f"no raw results under {RESULTS / 'raw' / tag / split}")
    for key in ("config_hash", "git_commit"):
        if len(vals := {r[key] for r in recs}) > 1 and not force:
            raise SystemExit(f"raw results have different {key}s {sorted(vals)}; rerun them or pass --force")
    return recs


def table(recs: list[dict], by: list[str], metrics: list[str], cfg: dict) -> pd.DataFrame:
    """One row per group: n_scenarios, then <metric>, <metric>_lo, <metric>_hi."""
    rows = []
    for key, grp in pd.DataFrame(recs).groupby(by, sort=True, dropna=False):
        agg = aggregate(list(grp["metrics"]), cfg)
        row = {**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n_scenarios": len(grp)}
        for m in metrics:
            a = agg.get(m, {"value": float("nan"), "lo": float("nan"), "hi": float("nan")})
            row |= {m: a["value"], f"{m}_lo": a["lo"], f"{m}_hi": a["hi"]}
        rows.append(row)
    return pd.DataFrame(rows)


def summary_md(recs: list[dict], main: pd.DataFrame, tag: str, split: str, errors: int) -> str:
    fmt = lambda r, m: "-" if pd.isna(r[m]) else f"{r[m]:.3f} [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}]"
    out = [f"# Results: tag `{tag}`, split `{split}`", "",
           f"- config hash: {', '.join(sorted({r['config_hash'] for r in recs}))}",
           f"- git commit: {', '.join(sorted({r['git_commit'] for r in recs}))}",
           f"- scenarios per method: {dict(zip(main['method'], main['n_scenarios']))}",
           f"- unresolved errors (errors.jsonl, no result file): {errors}"]
    if main["n_scenarios"].nunique() > 1:
        out.append("- **WARNING: methods cover different numbers of scenarios; they are not directly comparable.**")
    out += ["", "Pooled value [95% bootstrap CI].", "",
            "| method | " + " | ".join(SUMMARY) + " |", "|---" * (len(SUMMARY) + 1) + "|"]
    out += [f"| {r['method']} | " + " | ".join(fmt(r, m) for m in SUMMARY) + " |" for _, r in main.iterrows()]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", required=True)
    ap.add_argument("--force", action="store_true", help="merge results with different config hashes / git commits")
    args = ap.parse_args(argv)
    recs, cfg = load(args.tag, args.split, args.force), load_config()
    out = RESULTS / args.tag / args.split
    out.mkdir(parents=True, exist_ok=True)
    main_df = table(recs, ["method"], MAIN, cfg)
    main_df.to_csv(out / "main.csv", index=False)
    table(recs, ["method", "category"], MAIN, cfg).to_csv(out / "per_category.csv", index=False)
    pers = table(recs, ["method"], PERSISTENCE, cfg)
    pers.to_csv(out / "persistence.csv", index=False)
    cost = table(recs, ["method"], COST, cfg)
    fallbacks = pd.DataFrame(recs).groupby("method")["llm_stats"].apply(
        lambda s: sum(sum(x.get("fallbacks", {}).values()) for x in s))
    cost["llm_fallbacks"] = cost["method"].map(fallbacks)
    cost.to_csv(out / "cost.csv", index=False)
    err_file = RESULTS / "raw" / args.tag / "errors.jsonl"
    done = {(r["method"], r["scenario"]) for r in recs}  # errors.jsonl is append-only: drop ones a rerun fixed
    errs = [json.loads(l) for l in err_file.open(encoding="utf-8")] if err_file.exists() else []
    errors = len({(e["method"], e["scenario"]) for e in errs if e["split"] == args.split} - done)
    (out / "summary.md").write_text(summary_md(recs, main_df.merge(pers.filter(regex="^(method|persistence)(_lo|_hi)?$")), args.tag, args.split, errors), encoding="utf-8")
    print(f"wrote {out}/: main.csv per_category.csv persistence.csv cost.csv summary.md")


if __name__ == "__main__":
    main()
