"""Counterfactual repair-set selection (Spec 10; Paper IV-D, Section V).

Probes are generated from Delta + candidate memories only; gold labels are never read here.
"""
from __future__ import annotations

import hashlib
import json
import re

from csmr import llm
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline.actions import condition_text, revise_text
from csmr.pipeline.agent import answer
from csmr.pipeline.overlay import view
from csmr.types import MODIFYING_ACTIONS, Action, CandidateInfo, Memory, Observation, RepairSet, StateChange

PROBE_PROMPT = """Write neutral questions about the user's CURRENT state, phrased as a user or a downstream task would
ask them (e.g. "How does the user commute to work?", "Does the user own a car?", "Does the user prefer driving?").
Always phrase them in the third person ("the user"), never "you" or "your".
Do NOT mention or presuppose any recent change or event (no "now that...", "after selling...", "since you...").
Write one question per stored memory below (even if it seems unaffected) plus one question about the attribute
"{attribute}". Questions must concern only those memories or that attribute, never unrelated topics.

Stored memories:
{memories}

Return JSON only: {{"change_question": "...", "memory_questions": {{"<memory id>": "..."}}}}"""

JUDGE_PROMPT = """New fact ({time}): "{obs}" ({statement})

Question: {question}
Assistant answer: {answer}

Decide in two steps.
1. determined_by_change: does the new fact settle the correct answer to this question (true), or is the question
   unaffected / still undeterminable given the new fact (false)?
2. verdict:
   - "consistent" (1.0): the answer is correct given the new fact. If determined_by_change is false, "unknown" or an
     uncertain answer is consistent.
   - "vague" (0.5): determined_by_change is true but the answer is "unknown" or vague.
   - "contradicts" (0.0): the answer asserts something the new fact contradicts.
Return JSON only: {{"determined_by_change": true | false, "verdict": "consistent" | "vague" | "contradicts"}}"""
RETRY_NUDGE = "\nReturn exactly the JSON object described above."  # changes the prompt so the disk cache is not hit again
PROBE_FALLBACK = "Is it still true that {text}?"

VERDICT_SCORE = {"consistent": 1.0, "vague": 0.5, "contradicts": 0.0}


def _presupposes(q: str, obs: Observation, cands: dict[str, CandidateInfo], store: MemoryStoreAdapter) -> bool:
    """True if q uses a word only the observation introduces (e.g. "sold"). ponytail: crude lexical check, no stemming."""
    known = {w for m in cands for w in re.findall(r"[a-z]+", store.get(m).text.lower())}
    novel = {w for w in re.findall(r"[a-z]+", obs.text.lower()) if len(w) > 3 and w not in known}
    return any(w in novel for w in re.findall(r"[a-z]+", q.lower()))


def generate_probes(delta: StateChange, obs: Observation, cands: dict[str, CandidateInfo],
                    store: MemoryStoreAdapter, n: int) -> list[str]:
    """One neutral probe about Delta's attribute, then one per candidate (roots/closest hops first, KEEP included),
    presupposing probes dropped, capped at n."""
    mems = "\n".join(f"- {m}: {store.get(m).text}" for m in cands)
    out = llm.try_json(PROBE_PROMPT.format(attribute=delta.attribute.replace("_", " "), memories=mems),
                       max_tokens=load_config()["probe_max_tokens"])
    order = sorted(cands, key=lambda m: (not cands[m].is_root, cands[m].hop, m))
    if (out is None or not str(out.get("change_question") or "").strip()
            or not isinstance(out.get("memory_questions"), dict)):
        llm.fallback("probes", "malformed response; one template probe per candidate")
        qs = [PROBE_FALLBACK.format(text=store.get(m).text.rstrip(". ")) for m in order]
    else:
        per_mem = out["memory_questions"]
        qs = [str(out["change_question"]).strip()] + [str(per_mem[m]).strip() for m in order if m in per_mem]
    return [q for q in qs if not _presupposes(q, obs, cands, store)][:n]


def _unknown_like(ans: str, cfg: dict) -> bool:
    """Empty, or matches an unknown pattern and is short (<= unknown_max_words) or starts with the pattern."""
    a = ans.strip().lower()
    if not a:
        return True
    short = len(a.split()) <= cfg["unknown_max_words"]
    return any(re.search(p, a) if short else re.match(p, a) for p in cfg["unknown_patterns"])


def judge(ans: str, question: str, delta: StateChange, obs: Observation, cfg: dict) -> dict:
    """Two-step judge in one call; returns {"determined_by_change", "verdict", "score"} (Spec 10).
    Unknown-like answers get their verdict from determined_by_change in code, not from the LLM.
    A malformed response is retried once; if still malformed the verdict is "vague" (determined_by_change None)."""
    unknown = _unknown_like(ans, cfg)
    prompt = JUDGE_PROMPT.format(time=obs.time, obs=obs.text, statement=delta.statement or obs.text,
                                 question=question, answer=ans)
    for p in (prompt, prompt + RETRY_NUDGE):
        out = llm.try_json(p, model=cfg["llm"]["judge_model"], max_tokens=cfg["judge_max_tokens"])
        if out is not None and isinstance(out.get("determined_by_change"), bool) and (
                unknown or str(out.get("verdict")).strip().lower() in VERDICT_SCORE):
            break
    else:
        llm.fallback("judge", f"malformed response for {question!r}; verdict vague")
        return {"determined_by_change": None, "verdict": "vague", "score": VERDICT_SCORE["vague"]}
    determined = out["determined_by_change"]
    if unknown:
        verdict = "vague" if determined else "consistent"  # set in code; the LLM verdict is only for claims
    else:
        verdict = str(out["verdict"]).strip().lower()
    return {"determined_by_change": determined, "verdict": verdict, "score": VERDICT_SCORE[verdict]}


def prepare_texts(repair_sets: list[RepairSet], store: MemoryStoreAdapter, delta: StateChange,
                  obs: Observation) -> tuple[dict[str, str], dict[str, str]]:
    """REVISE texts and CONDITIONAL conditions needed by any set (one LLM call per memory; llm.call caches)."""
    revised: dict[str, str] = {}
    conditions: dict[str, str] = {}
    for rs in repair_sets:
        for mid, a in rs.actions.items():
            if a == Action.REVISE and mid not in revised:
                revised[mid] = revise_text(store.get(mid), delta, obs)
            elif a == Action.CONDITIONAL and mid not in conditions:
                conditions[mid] = condition_text(store.get(mid), delta, obs)
    return revised, conditions


def _evaluate(rs: RepairSet, probes: list[str], store: MemoryStoreAdapter, revised: dict[str, str],
              conditions: dict[str, str], cands: dict[str, CandidateInfo], delta: StateChange, obs: Observation,
              cfg: dict, cache: dict, new_mem: Memory | None) -> tuple[dict[str, float], list[dict]]:
    mems = view(store, rs, revised, conditions, new_mem)
    vhash = hashlib.sha256(json.dumps([(m.id, m.text) for m in mems]).encode()).hexdigest()
    details = []
    for q in probes:  # identical views reuse answers/judgements
        if (q, vhash) not in cache:
            a = answer(q, mems)
            cache[(q, vhash)] = {"question": q, "answer": a, **judge(a, q, delta, obs, cfg)}
        details.append(cache[(q, vhash)])
    vals = [d["score"] for d in details]
    benefit = sum(vals) / len(vals) if vals else 0.0
    collateral = sum(1 - cands[m].p_dep for m, a in rs.actions.items() if a in MODIFYING_ACTIONS)
    rewrites = sum(a in (Action.REVISE, Action.CONDITIONAL) for a in rs.actions.values())  # one rewrite LLM call each
    evidence_cost = (sum(a == Action.VERIFY for a in rs.actions.values()) * cfg["verify_cost"]
                     + rewrites * cfg["call_cost"])
    lc, ls = (cfg["lambda_c"], cfg["lambda_s"]) if cfg["ablation"]["use_minimality"] else (0.0, 0.0)
    size = rs.size()
    total = benefit - lc * collateral - cfg["lambda_e"] * evidence_cost - ls * size
    return ({"benefit": benefit, "collateral": collateral, "evidence_cost": evidence_cost, "size": size,
             "total": total}, details)


def score(rs: RepairSet, probes: list[str], store: MemoryStoreAdapter, revised: dict[str, str],
          conditions: dict[str, str], cands: dict[str, CandidateInfo], delta: StateChange, obs: Observation,
          cfg: dict, cache: dict | None = None, new_mem: Memory | None = None) -> dict[str, float]:
    """total = benefit - lc*collateral - le*evidence_cost - ls*|R|  (Section V)."""
    cache = {} if cache is None else cache
    return _evaluate(rs, probes, store, revised, conditions, cands, delta, obs, cfg, cache, new_mem)[0]


def select(repair_sets: list[RepairSet], probes: list[str], store: MemoryStoreAdapter, revised: dict[str, str],
           conditions: dict[str, str], cands: dict[str, CandidateInfo], delta: StateChange, obs: Observation,
           cfg: dict, new_mem: Memory | None = None) -> tuple[RepairSet, list[dict]]:
    """argmax of total over feasible sets, ties -> smaller |R|. Returns (chosen, per-set trace rows)."""
    feasible = [rs for rs in repair_sets if rs.feasible]
    details: dict[str, list[dict]] = {}  # per-set probe answers + judge fields, for the trace
    if cfg["ablation"]["use_counterfactual"]:
        cache: dict = {}
        for rs in feasible:
            rs.score, details[rs.name] = _evaluate(rs, probes, store, revised, conditions, cands, delta, obs, cfg,
                                                   cache, new_mem)
        top = max(r.score["total"] for r in feasible)  # near-ties (within tie_eps) fall through to smaller |R|
        best = min((r for r in feasible if r.score["total"] >= top - cfg["tie_eps"]), key=lambda r: r.score["size"])
    else:  # ablation: apply all proposed actions (R3; dedup may have renamed it to a lower-index twin)
        best = [rs for rs in feasible if rs.name <= "R3"][-1]
    rows = [{"name": rs.name, "actions": {k: v.value for k, v in rs.actions.items()}, "feasible": rs.feasible,
             "infeasible_reason": rs.infeasible_reason, "score": rs.score, "judgements": details.get(rs.name, []), "chosen": rs is best}
            for rs in repair_sets]
    return best, rows
