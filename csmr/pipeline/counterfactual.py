"""Counterfactual repair-set selection (Spec 10; Paper IV-D, Section V).

Probes are generated from candidate memory texts only (root text for the change question); gold labels are never read here.
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
ask them (e.g. "What does the user grow in their garden?").
Ask open questions (what / which / where / how / who), never yes/no questions. Never use change-hinting words such as
"still", "anymore", "no longer", "now" or "yet".
Each question must be the question its memory (for the change_question: the fact) directly answers, phrased so it does
NOT assume the memory's claim is true: ask about the memory's main predicate, not a detail that presupposes it.
E.g. memory "The user grows tomatoes in their garden": good "What does the user grow in their garden?";
bad "Which tomato variety does the user grow?" (presupposes tomatoes).
Always phrase them in the third person ("the user"), never "you" or "your".
Name the specific subject exactly as the memory text names it (a project, product, place or person, e.g.
"Which theme does the Fernhill wiki use?"), never a generic stand-in such as "the project" or a field name.
Do NOT mention or presuppose any recent change or event (no "now that...", "after switching...", "since you...").
Write one question per stored memory below (even if it seems unaffected) plus one "change_question" asking for the
current value of this stored fact: "{fact}"
Questions must concern only those memories or that fact, never unrelated topics.

Stored memories:
{memories}

Return JSON only: {{"change_question": "...", "memory_questions": {{"<memory id>": "..."}}}}"""

JUDGE_PROMPT = """New fact ({time}): "{obs}" ({statement})
Stored memory the question was written from: "{source}"

Question: {question}
Assistant answer: {answer}

Decide in two steps.
1. determined_by_change: does the new fact settle the correct answer to this question (true), or is the question
   unaffected / still undeterminable given the new fact (false)?
2. verdict. Only claims about the CURRENT state can contradict: mentioning an old value explicitly as past or
   previous ("it used X before; the current method is unknown", "previously had X") is NOT asserting it.
   - "contradicts" (0.0): the answer asserts something the new fact contradicts OR makes likely outdated, as the
     current state (e.g. it still states a fact that held only because of the old state).
     It is also "contradicts" if the answer still asserts the part of the stored memory that the new fact makes false
     or likely outdated, even when the answer leaves out the specific detail (e.g. stored memory "The user buys
     sourdough from a bakery in Lisbon", new fact "The user left Lisbon and lives in Oslo", answer "The user buys
     sourdough from a bakery." -> contradicts).
     This applies whatever determined_by_change is.
   - "consistent" (1.0): the answer is correct given the new fact. If determined_by_change is false, "unknown" or an
     uncertain answer is consistent.
   - "vague" (0.5): determined_by_change is true but the answer is "unknown" or vague.
Return JSON only: {{"determined_by_change": true | false, "verdict": "consistent" | "vague" | "contradicts"}}"""
RETRY_NUDGE = "\nReturn exactly the JSON object described above."  # changes the prompt so the disk cache is not hit again
PROBE_FALLBACK = "What is the current state of this: {text}?"

VERDICT_SCORE = {"consistent": 1.0, "vague": 0.5, "contradicts": 0.0}


def _presupposes(q: str, obs: Observation, cands: dict[str, CandidateInfo], store: MemoryStoreAdapter) -> bool:
    """True if q uses a word only the observation introduces (e.g. "sold"). ponytail: crude lexical check, no stemming."""
    known = {w for m in cands for w in re.findall(r"[a-z]+", store.get(m).text.lower())}
    novel = {w for w in re.findall(r"[a-z]+", obs.text.lower()) if len(w) > 3 and w not in known}
    return any(w in novel for w in re.findall(r"[a-z]+", q.lower()))


def _change_hinting(q: str, cfg: dict) -> bool:
    """True if q is a yes/no question (starts with an auxiliary) or contains a change-hinting word ("still")."""
    words = re.findall(r"[a-z]+", q.lower())
    return (not words or words[0] in cfg["probe_yes_no_starts"]
            or any(re.search(p, q.lower()) for p in cfg["probe_banned_patterns"]))


def _norm(q: str) -> str:
    """Duplicate key: case- and whitespace-insensitive."""
    return " ".join(q.casefold().split())


def _pick(qs: list[tuple[str, str | None]], cands: dict[str, CandidateInfo], store: MemoryStoreAdapter,
          n: int) -> list[dict]:
    """First probe of every candidate proposed for a non-KEEP action (even beyond n), then the rest in order up to n."""
    must: dict[str, int] = {}
    for i, (_, m) in enumerate(qs):
        if m and cands[m].proposed != Action.KEEP:
            must.setdefault(m, i)
    rest = [i for i in range(len(qs)) if i not in must.values()][:max(n - len(must), 0)]
    return [{"question": q, "source": m, "source_text": store.get(m).text if m else ""}
            for q, m in (qs[i] for i in sorted([*must.values(), *rest]))]


def generate_probes(delta: StateChange, obs: Observation, cands: dict[str, CandidateInfo],
                    store: MemoryStoreAdapter, n: int, cfg: dict | None = None) -> tuple[list[dict], dict[str, str]]:
    """One neutral probe about the root fact's current value (first candidate in order if there is no root), then
    one per candidate (roots/closest hops first, KEEP included); presupposing, yes/no, change-hinting and duplicate
    probes dropped; every candidate proposed for a non-KEEP action keeps one probe, other slots fill in order up to n.
    Returns (probes, gaps): probes are {"question", "source", "source_text"} (source = the memory the probe was written
    from; the change probe's is the root); gaps maps a non-KEEP-proposed candidate left without a probe to
    "missing" | "filtered" | "duplicate".
    The attribute key is never shown: probes name the subject as the memory text does."""
    cfg = cfg or load_config()
    mems = "\n".join(f"- {m}: {store.get(m).text}" for m in cands)
    order = sorted(cands, key=lambda m: (not cands[m].is_root, -cands[m].p_dep, int(re.sub(r"\D", "", m) or 0), m))
    fact_ids = [m for m in order if cands[m].is_root] or order[:1]
    fact = " ".join(store.get(m).text for m in fact_ids)
    out = llm.try_json(PROBE_PROMPT.format(fact=fact, memories=mems), max_tokens=cfg["probe_max_tokens"], cfg=cfg)
    # templates are built from memory text only, so they are never filtered (a "now"/"still" there is the memory's own)
    templates = [(PROBE_FALLBACK.format(text=store.get(m).text.rstrip(". ")), m) for m in order]
    templates = [t for i, t in enumerate(templates) if _norm(t[0]) not in {_norm(u[0]) for u in templates[:i]}]
    if (out is None or not str(out.get("change_question") or "").strip()
            or not isinstance(out.get("memory_questions"), dict)):
        llm.fallback("probes", "malformed response; one template probe per candidate")
        return _pick(templates, cands, store, n), {}
    per_mem = out["memory_questions"]
    lost = {m: "missing" for m in order if m not in per_mem}
    kept: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    raw = [(str(out["change_question"]).strip(), fact_ids[0] if fact_ids else None)]
    for q, m in raw + [(str(per_mem[m]).strip(), m) for m in order if m in per_mem]:
        key = _norm(q)
        if _presupposes(q, obs, cands, store) or _change_hinting(q, cfg):
            lost.setdefault(m, "filtered")
        elif key in seen:
            lost.setdefault(m, "duplicate")
        else:
            seen.add(key)
            kept.append((q, m))
    if not kept:
        llm.fallback("probes", "every probe filtered out; one template probe per candidate")
        return _pick(templates, cands, store, n), {}
    probes = _pick(kept, cands, store, n)
    covered = {p["source"] for p in probes}
    gaps = {m: lost[m] for m in order if cands[m].proposed != Action.KEEP and m not in covered}
    return probes, gaps


def _unknown_like(ans: str, cfg: dict) -> bool:
    """Empty, or matches an unknown pattern and is short (<= unknown_max_words) or starts with the pattern."""
    a = ans.strip().lower()
    if not a:
        return True
    short = len(a.split()) <= cfg["unknown_max_words"]
    return any(re.search(p, a) if short else re.match(p, a) for p in cfg["unknown_patterns"])


def judge(ans: str, question: str, delta: StateChange, obs: Observation, cfg: dict, source_text: str = "") -> dict:
    """Two-step judge in one call; returns {"determined_by_change", "verdict", "score"} (Spec 10).
    source_text: the (pre-commit) memory the probe was written from, so a stale claim is caught even if the answer
    drops the outdated detail.
    Unknown-like answers get their verdict from determined_by_change in code, not from the LLM.
    A malformed response is retried once; if still malformed the verdict is "vague" (determined_by_change None)."""
    unknown = _unknown_like(ans, cfg)
    prompt = JUDGE_PROMPT.format(time=obs.time[:10], obs=obs.text, statement=delta.statement or obs.text,
                                 source=source_text or "(none)", question=question, answer=ans)
    for p in (prompt, prompt + RETRY_NUDGE):
        out = llm.try_json(p, model=cfg["llm"]["judge_model"], max_tokens=cfg["judge_max_tokens"], cfg=cfg)
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
                  obs: Observation, cfg: dict | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """REVISE texts and CONDITIONAL conditions needed by any set (one LLM call per memory; llm.call caches)."""
    revised: dict[str, str] = {}
    conditions: dict[str, str] = {}
    for rs in repair_sets:
        for mid, a in rs.actions.items():
            if a == Action.REVISE and mid not in revised:
                revised[mid] = revise_text(store.get(mid), delta, obs, cfg)
            elif a == Action.CONDITIONAL and mid not in conditions:
                conditions[mid] = condition_text(store.get(mid), delta, obs, cfg)
    return revised, conditions


def _evaluate(rs: RepairSet, probes: list[dict], store: MemoryStoreAdapter, revised: dict[str, str],
              conditions: dict[str, str], cands: dict[str, CandidateInfo], delta: StateChange, obs: Observation,
              cfg: dict, cache: dict, new_mem: Memory | None) -> tuple[dict[str, float], list[dict]]:
    mems = view(store, rs, revised, conditions, new_mem)
    vhash = hashlib.sha256(json.dumps([(m.id, m.text) for m in mems]).encode()).hexdigest()
    details = []
    for p in probes:  # identical views reuse answers/judgements
        q, key = p["question"], (p["question"], p["source_text"], vhash)
        if key not in cache:
            a = answer(q, mems, cfg=cfg)
            cache[key] = {"question": q, "source": p["source"], "answer": a,
                          **judge(a, q, delta, obs, cfg, p["source_text"])}
        details.append(cache[key])
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


def score(rs: RepairSet, probes: list[dict], store: MemoryStoreAdapter, revised: dict[str, str],
          conditions: dict[str, str], cands: dict[str, CandidateInfo], delta: StateChange, obs: Observation,
          cfg: dict, cache: dict | None = None, new_mem: Memory | None = None) -> dict[str, float]:
    """total = benefit - lc*collateral - le*evidence_cost - ls*|R|  (Section V)."""
    cache = {} if cache is None else cache
    return _evaluate(rs, probes, store, revised, conditions, cands, delta, obs, cfg, cache, new_mem)[0]


def select(repair_sets: list[RepairSet], probes: list[dict], store: MemoryStoreAdapter, revised: dict[str, str],
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
    else:  # ablation (SPEC §10): R3 = apply all proposed actions; dedup may have renamed it to a lower-index twin
        r3 = [rs for rs in repair_sets if rs.name <= "R3"][-1]
        # hard rules still bind: an infeasible R3 (weak-evidence root forced to REVISE) falls back to the largest feasible lower set
        best = r3 if r3.feasible else [rs for rs in feasible if rs.name < r3.name][-1]
    rows = [{"name": rs.name, "actions": {k: v.value for k, v in rs.actions.items()}, "feasible": rs.feasible,
             "infeasible_reason": rs.infeasible_reason, "score": rs.score, "judgements": details.get(rs.name, []), "chosen": rs is best}
            for rs in repair_sets]
    return best, rows
