"""Evaluation metrics (Spec 14; Paper VIII). Reads gold labels: allowed only here and in fit() code.

Every metric maps one Record to {name: (numerator, denominator)}. Its value over a set of scenarios is
sum(numerators) / sum(denominators), i.e. pooled over memories / questions, and `aggregate` puts a scenario-level
bootstrap confidence interval on that ratio."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from csmr import llm
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline.agent import answer
from csmr.pipeline.overlay import render
from csmr.scenario import Scenario
from csmr.types import MODIFYING_ACTIONS, Action, Memory, Trace

Counts = dict[str, tuple[float, float]]

# Not the selection judge (counterfactual.JUDGE_PROMPT): this one grades against the gold answer and never sees Δ.
QA_JUDGE_PROMPT = """You are grading an assistant's answer against a reference answer.

Question: {question}
Reference answer: {gold}
Assistant answer: {answer}

The assistant answer is correct if it gives the same information about the CURRENT state as the reference.
- Wording, extra detail and expressed uncertainty do not matter, as long as nothing in it conflicts with the reference.
- If the reference says the answer is unknown or that something no longer holds, the assistant answer is correct only
  if it does not assert a current value that the reference rules out.
- If the reference gives a definite answer, "unknown" is incorrect.
Return JSON only: {{"correct": true | false}}"""


@dataclass
class Record:
    """One method's result on one scenario."""
    scenario: Scenario
    final_actions: dict[str, str]              # memory id -> action; ids left out are KEEP
    trace: Trace | None = None                 # None for a method with no CSMR trace: trace metrics are skipped
    store: MemoryStoreAdapter | None = None    # committed store; only QA accuracy and persistence need it


def _gold(rec: Record) -> dict[str, Action]:
    """Gold action per memory. Raises on an unlabeled memory or a label outside the five actions."""
    return {m.id: Action(rec.scenario.gold.actions[m.id]) for m in rec.scenario.memories}


def repair(rec: Record) -> Counts:
    """Repair precision / recall, over-invalidation, under-repair, 5-way action accuracy (Paper VIII).
    changed = REVISE/REVOKE. Under-repair counts only gold-changed memories left KEEP, so it is not 1 - recall."""
    gold = _gold(rec)
    pred = {i: Action(rec.final_actions.get(i, Action.KEEP)) for i in gold}
    changed = {i for i, a in pred.items() if a in MODIFYING_ACTIONS}
    gold_changed = {i for i, a in gold.items() if a in MODIFYING_ACTIONS}
    gold_keep = {i for i, a in gold.items() if a == Action.KEEP}
    tp = len(changed & gold_changed)
    return {"repair_precision": (tp, len(changed)),
            "repair_recall": (tp, len(gold_changed)),
            "over_invalidation": (len(changed & gold_keep), len(gold_keep)),
            "under_repair": (sum(pred[i] == Action.KEEP for i in gold_changed), len(gold_changed)),
            "action_accuracy": (sum(pred[i] == gold[i] for i in gold), len(gold))}


def dependency(rec: Record, cfg: dict) -> Counts:
    """Dependent-level dependency P/R/F1 (Paper VIII). The classifier scores each non-root candidate against the roots
    and the trace keeps the max, so a gold (root, dependent) pair is matched on its dependent: predicted = non-root
    candidates with P_dep >= tau_d, gold = dependents in gold.dependencies. Roots (P_dep = 1 by construction) are left
    out of both."""
    if rec.trace is None:
        return {}
    cands = rec.trace.candidates
    roots = {i for i, c in cands.items() if c["is_root"]}
    pred = {i for i, c in cands.items() if c["p_dep"] >= cfg["tau_d"]} - roots
    gold = {d for _, d in rec.scenario.gold.dependencies} - roots
    tp = len(pred & gold)
    return {"dependent_precision": (tp, len(pred)), "dependent_recall": (tp, len(gold)),
            "dependent_f1": (2 * tp, len(pred) + len(gold))}


def candidate_recall(rec: Record) -> Counts:
    """Share of gold non-KEEP memories that reached the candidate set C (Spec 14)."""
    if rec.trace is None:
        return {}
    need = {i for i, a in _gold(rec).items() if a != Action.KEEP}
    return {"candidate_recall": (len(need & set(rec.trace.candidates)), len(need))}


def qa_judge(question: str, gold_answer: str, ans: str, cfg: dict) -> bool:
    """LLM grade of one answer against the gold answer. A malformed response is retried once (try_json_valid, with a
    changed prompt); if still malformed it counts as incorrect and as a "qa_judge" fallback."""
    out = llm.try_json_valid(QA_JUDGE_PROMPT.format(question=question, gold=gold_answer, answer=ans),
                             lambda o: isinstance(o.get("correct"), bool),
                             model=cfg["llm"]["judge_model"], max_tokens=cfg["judge_max_tokens"], cfg=cfg)
    if out is None:
        llm.fallback("qa_judge", f"malformed response for {question!r}; counted incorrect")
        return False
    return out["correct"]


def _qa(rec: Record, cfg: dict, later: list[Memory] = ()) -> tuple[int, int]:
    mems = [r for m in rec.store.all() if (r := render(m))] + list(later)  # what the agent sees after the commit
    qs = rec.scenario.gold.eval_questions
    return sum(qa_judge(e["q"], e["gold_answer"], answer(e["q"], mems, cfg=cfg), cfg) for e in qs), len(qs)


def qa(rec: Record, cfg: dict) -> Counts:
    """Downstream QA accuracy: the shared agent answers the gold eval questions on the committed store (Paper VIII)."""
    return {} if rec.store is None else {"qa_accuracy": _qa(rec, cfg)}


def persistence(rec: Record, cfg: dict) -> Counts:
    """Persistence of correction (Paper VIII): QA accuracy after later-turn memories are appended to the agent's view
    (never to the store). Uses the scenario's gold.persistence_memories, which must be dated after the observation;
    a scenario without them falls back to cfg.persistence_distractors, counted in "persistence_fallback"."""
    if rec.store is None:
        return {}
    seen = datetime.fromisoformat(rec.scenario.observation.time)
    later = rec.scenario.gold.persistence_memories
    fallback = not later
    if fallback:
        later = [Memory(id=f"DIS{n}", text=t, entity="user", attribute="distractor",
                        time=(seen + timedelta(days=n)).isoformat())  # any later time: only the order matters
                 for n, t in enumerate(cfg["persistence_distractors"], 1)]
    elif early := [m.id for m in later if datetime.fromisoformat(m.time) <= seen]:
        raise ValueError(f"{rec.scenario.id}: persistence_memories not dated after the observation: {early}")
    return {"persistence": _qa(rec, cfg, later), "persistence_fallback": (int(fallback), 1)}


def cost(rec: Record) -> Counts:
    """LLM calls (cache hits included, so reruns agree) per scenario, and wall time from uncached runs only: a run
    that was served any reply cached by an earlier run (llm_stats.prior_hits) reports no wall time (Spec 14)."""
    if rec.trace is None:
        return {}
    stats = rec.trace.llm_stats
    out = {"llm_calls": (stats.get("calls", 0), 1)}
    if stats.get("prior_hits") == 0:  # absent (trace from before the counter existed) = unknown = excluded
        out["wall_time_s"] = (rec.trace.wall_time_s, 1)
    return out


def scenario_metrics(rec: Record, cfg: dict | None = None) -> Counts:
    """Every metric for one record."""
    cfg = cfg or load_config()
    return {**repair(rec), **dependency(rec, cfg), **candidate_recall(rec), **qa(rec, cfg), **persistence(rec, cfg),
            **cost(rec)}


def aggregate(rows: list[Counts], cfg: dict | None = None) -> dict[str, dict[str, float]]:
    """Pool each metric over scenarios and add a scenario-level percentile bootstrap CI: whole scenarios are
    resampled with replacement and the pooled ratio recomputed. All metrics share the same resamples.
    Returns {metric: {"value", "lo", "hi", "n"}}; n = scenarios where the metric is defined (denominator > 0)."""
    cfg = cfg or load_config()
    idx = np.random.default_rng(cfg["seed"]).integers(0, len(rows), (cfg["bootstrap_samples"], len(rows)))
    tail = 100 * (1 - cfg["ci_level"]) / 2
    out = {}
    for name in dict.fromkeys(k for r in rows for k in r):
        nd = np.array([r.get(name, (0, 0)) for r in rows], dtype=float)  # scenarios x (num, den)
        tot = nd[idx].sum(axis=1)
        with np.errstate(invalid="ignore"):  # 0/0 = undefined (e.g. nothing changed in a whole resample)
            boot = tot[:, 0] / tot[:, 1]
            value = nd[:, 0].sum() / nd[:, 1].sum()
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [tail, 100 - tail]) if boot.size else (np.nan, np.nan)
        out[name] = {"value": float(value), "lo": float(lo), "hi": float(hi), "n": int((nd[:, 1] > 0).sum())}
    return out
