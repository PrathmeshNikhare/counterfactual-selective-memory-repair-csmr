"""Proposed-action rule and REVISE / CONDITIONAL text generation (Paper IV-C, IV-F)."""
from __future__ import annotations

from csmr import llm
from csmr.types import Action, CandidateInfo, Memory, Observation, StateChange

REVISE_PROMPT = """A stored memory may be outdated because of a new observation. Rewrite the memory so it stays
true: keep what was true before (as past), and say what is now unknown or changed. One sentence, third person.
Example: "The user previously drove to work by car; current commute mode unknown after selling the car (as of 2026-06-01)."

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


def propose_action(mid: str, info: CandidateInfo | None, cfg: dict, change_type: str = "replace") -> Action:
    """Paper IV-C rule; info is None when mid is not in the candidate set C (then KEEP)."""
    if info is None:
        return Action.KEEP
    strong = info.s_evidence >= cfg["tau_e"]
    dependent = info.p_dep >= cfg["tau_d"]
    if info.is_root and strong:
        return Action.REVOKE if change_type == "remove" else Action.REVISE
    if dependent and strong:
        return Action.REVISE
    if dependent and info.temporary:
        return Action.CONDITIONAL
    return Action.VERIFY if dependent else Action.KEEP


def _ask(prompt: str, key: str, mem: Memory, delta: StateChange, obs: Observation) -> str:
    out = llm.call(prompt.format(memory=mem.text, time=obs.time, obs=obs.text, statement=delta.statement or obs.text),
                   json_mode=True)
    return str(out[key]).strip()


def revise_text(mem: Memory, delta: StateChange, obs: Observation) -> str:
    return _ask(REVISE_PROMPT, "text", mem, delta, obs)


def condition_text(mem: Memory, delta: StateChange, obs: Observation) -> str:
    return _ask(CONDITION_PROMPT, "condition", mem, delta, obs)
