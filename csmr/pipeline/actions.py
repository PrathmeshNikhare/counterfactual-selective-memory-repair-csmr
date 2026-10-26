"""Proposed-action rule and REVISE / CONDITIONAL text generation (Paper IV-C, IV-F)."""
from __future__ import annotations

from csmr import llm
from csmr.config import load_config
from csmr.types import Action, CandidateInfo, Memory, Observation, StateChange

# Examples must never mirror dev/test/showcase scenarios.
REVISE_PROMPT = """A stored memory may be outdated because of a new observation. Rewrite it as ONE short sentence,
third person, present tense, that states the CURRENT state first; put the old value only in brackets at the end.
If the current value is not known from the observation, say it is unknown. Do not include any date.
Examples:
- the fact itself changed: "The user's blog is hosted on Ghost (moved from WordPress)."
- a fact that depended on it, new value unknown: "The user's current blog theme is unknown (it used the Astra theme
  before the move to Ghost)."

Memory: "{memory}"
New observation ({time}): "{obs}"
Change: {statement}
Return JSON only: {{"text": "<rewritten memory>"}}"""

CONDITION_PROMPT = """A stored memory is affected by a TEMPORARY change. Write the condition under which the memory
holds (not a rewrite of the memory). One short clause, e.g. "while the iPhone is not at the service centre".

Memory: "{memory}"
New observation ({time}): "{obs}"
Change: {statement}
Return JSON only: {{"condition": "<condition clause>"}}"""
REVISE_FALLBACK = "Previously: {text}. Possibly outdated after: {statement}"
CONDITION_FALLBACK = "it is not affected by: {statement}"


def propose_action(mid: str, info: CandidateInfo | None, cfg: dict, change_type: str = "replace") -> Action:
    """Paper IV-C rule; info is None when mid is not in the candidate set C (then KEEP)."""
    if info is None:
        return Action.KEEP
    strong = info.s_evidence >= cfg["tau_e"]
    dependent = info.p_dep >= cfg["tau_d"]
    if dependent and info.temporary:  # before REVISE/REVOKE, roots included (P_dep = 1)
        return Action.CONDITIONAL
    if info.is_root and strong:
        return Action.REVOKE if change_type == "remove" else Action.REVISE
    if dependent and strong:
        return Action.REVISE
    return Action.VERIFY if dependent else Action.KEEP


def _ask(prompt: str, key: str, mem: Memory, delta: StateChange, obs: Observation, cfg: dict | None) -> str | None:
    """Rewrite text for mem, or None (after one retry) if the response is malformed."""
    cfg = cfg or load_config()
    out = llm.try_json_valid(
        prompt.format(memory=mem.text, time=obs.time[:10], obs=obs.text, statement=delta.statement or obs.text),
        lambda o: bool(str(o.get(key) or "").strip()), max_tokens=cfg["rewrite_max_tokens"],
        cfg=cfg)
    return str(out[key]).strip() if out else None


def revise_text(mem: Memory, delta: StateChange, obs: Observation, cfg: dict | None = None) -> str:
    text = _ask(REVISE_PROMPT, "text", mem, delta, obs, cfg)
    if text is None:
        llm.fallback("revise_text", f"malformed response for {mem.id}; template text")
        text = REVISE_FALLBACK.format(text=mem.text.rstrip(". "), statement=delta.statement or obs.text)
    return text


def condition_text(mem: Memory, delta: StateChange, obs: Observation, cfg: dict | None = None) -> str:
    cond = _ask(CONDITION_PROMPT, "condition", mem, delta, obs, cfg)
    if cond is None:
        llm.fallback("condition_text", f"malformed response for {mem.id}; template condition")
        cond = CONDITION_FALLBACK.format(statement=delta.statement or obs.text)
    return cond
