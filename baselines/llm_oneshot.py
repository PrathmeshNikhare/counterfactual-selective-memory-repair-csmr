"""llm_oneshot (Spec 13): one JSON LLM call (through csmr/llm.py) with every memory and the observation, returning an
action per memory. Malformed output (after llm's one retry) falls back to all KEEP and counts an "llm_oneshot"
fallback. REVISE texts / CONDITIONAL conditions come from the same helpers CSMR uses."""
from __future__ import annotations

from baselines.common import run_baseline
from csmr import llm
from csmr.types import Action

NAME = "llm_oneshot"

ONESHOT_PROMPT = """You maintain an assistant's long-term memory about a user. A new observation has arrived. Decide, for
EVERY stored memory below, what to do with it now:
- KEEP: leave it unchanged.
- VERIFY: it may be outdated; mark it for verification before relying on it.
- CONDITIONAL: it stays valid only under a narrower condition.
- REVISE: its content must be updated.
- REVOKE: it no longer holds; deactivate it (its history is kept).

New observation ({time}, source: {source}): "{obs}"

Stored memories (id [date]: text):
{memories}

Return JSON only: {{"actions": {{"<memory id>": "KEEP" | "VERIFY" | "CONDITIONAL" | "REVISE" | "REVOKE"}}}}"""

_VALID = {a.value for a in Action}


def _ok(out: dict) -> bool:
    acts = out.get("actions")
    return isinstance(acts, dict) and all(str(v).strip().upper() in _VALID for v in acts.values())


def decide(delta, obs, roots, store, cfg):
    mems = store.all()
    prompt = ONESHOT_PROMPT.format(time=obs.time[:10], source=obs.source, obs=obs.text,
                                   memories="\n".join(f"- {m.id} [{m.time[:10] or 'undated'}]: {m.text}" for m in mems))
    out = llm.try_json_valid(prompt, _ok, max_tokens=cfg["oneshot_max_tokens"], cfg=cfg)
    if out is None:
        llm.fallback("llm_oneshot", "malformed response; every memory KEEP")
        return {}, {}
    ids = {m.id for m in mems}  # unknown ids are ignored; memories left out are KEEP
    return {m: Action(str(a).strip().upper()) for m, a in out["actions"].items() if m in ids}, {}


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
