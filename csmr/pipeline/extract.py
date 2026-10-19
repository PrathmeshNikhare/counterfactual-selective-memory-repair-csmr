"""Phase A — state-change extraction and direct conflict roots (Paper IV-A). Never modifies the store."""
from __future__ import annotations

from csmr import llm, models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.types import Observation, StateChange

# Few-shot examples must never mirror dev/test scenarios (no cars, bikes, jobs, cities, pets, phones).
EXTRACT_PROMPT = """You read one observation from a user talking to an assistant and decide whether it
reports a CHANGE in the state of the world that could make previously stored facts outdated.

Observation: "{text}"

Attribute keys already used in memory: {known}
Reuse one of those keys if it fits; otherwise invent a short snake_case key (e.g. vehicle_ownership, employment, residence).

Return JSON only.
Tentative, planned or hedged changes ("might", "planning to", "not sure") MUST still be extracted, with a lower
confidence; judging how weak the evidence is is not your job. Return null only if the message changes no stored
fact at all, i.e. nothing stored becomes false or outdated (a new preference, a plain fact with nothing to
replace): {{"state_change": null}}.
Otherwise return:
{{"state_change": {{"entity": "<who/what changed, e.g. user>", "attribute": "<snake_case key>",
"old": "<previous value or null>", "new": "<new value>", "confidence": <0..1, lower if hedged/uncertain>,
"statement": "<third-person restatement, e.g. The user sold their car.>",
"change_type": "remove" or "replace"}}}}
change_type:
- "remove" = the fact stops holding with no new value of the same kind
  (I cancelled my Netflix subscription, I stopped going to yoga classes)
- "replace" = a new value of the same kind takes its place
  (I switched my bank account to HDFC, My favourite team is now Mumbai Indians)"""


def extract_state_change(obs: Observation, known_attributes: list[str], cfg: dict | None = None) -> StateChange | None:
    """Paper IV-A: Δ = (entity, attribute, old, new, time, source, confidence). One LLM call."""
    prompt = EXTRACT_PROMPT.format(text=obs.text, known=sorted(set(known_attributes)) or "none")
    cfg = cfg or load_config()
    out = llm.try_json(prompt, max_tokens=cfg["extract_max_tokens"], cfg=cfg)
    if out is None or "state_change" not in out:
        llm.fallback("extract", "malformed response; treated as no state change")
        return None
    sc = out["state_change"]
    if not sc:
        return None
    try:
        return StateChange(entity=sc["entity"], attribute=sc["attribute"], old=sc.get("old"), new=str(sc["new"]),
                           time=obs.time, source=obs.source, confidence=float(sc["confidence"]),
                           statement=sc.get("statement") or obs.text,
                           change_type="remove" if sc.get("change_type") == "remove" else "replace")
    except (KeyError, TypeError, ValueError, AttributeError):
        llm.fallback("extract", f"missing/invalid fields in {sc!r}; treated as no state change")
        return None


def find_roots(delta: StateChange, obs: Observation, store: MemoryStoreAdapter, cfg: dict | None = None) -> list[str]:
    """Paper IV-A: same (entity, attribute); only if none, fall back to NLI contradiction(Δ.statement, m.text) >= threshold."""
    ms = store.all()
    roots = [m.id for m in ms if m.entity == delta.entity and m.attribute == delta.attribute]
    if roots:
        return roots
    thr = (cfg or load_config())["root_nli_threshold"]
    probs = models.nli_batch([(delta.statement, m.text) for m in ms])
    return [m.id for m, p in zip(ms, probs) if p["contradiction"] >= thr]
