"""Phase A — state-change extraction and direct conflict roots (Paper IV-A). Never modifies the store."""
from __future__ import annotations

from csmr import llm, models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.types import Observation, StateChange

EXTRACT_PROMPT = """You read one observation from a user talking to an assistant and decide whether it
reports a CHANGE in the state of the world that could make previously stored facts outdated.

Observation: "{text}"

Attribute keys already used in memory: {known}
Reuse one of those keys if it fits; otherwise invent a short snake_case key (e.g. vehicle_ownership, employment, residence).

Return JSON only.
If the observation reports no state change (a preference, a plain fact with nothing to replace), return {{"state_change": null}}.
Otherwise return:
{{"state_change": {{"entity": "<who/what changed, e.g. user>", "attribute": "<snake_case key>",
"old": "<previous value or null>", "new": "<new value>", "confidence": <0..1, lower if hedged/uncertain>,
"statement": "<third-person restatement, e.g. The user sold their car.>"}}}}"""


def extract_state_change(obs: Observation, known_attributes: list[str]) -> StateChange | None:
    """Paper IV-A: Δ = (entity, attribute, old, new, time, source, confidence). One LLM call."""
    prompt = EXTRACT_PROMPT.format(text=obs.text, known=sorted(set(known_attributes)) or "none")
    sc = llm.call(prompt, json_mode=True).get("state_change")
    if not sc:
        return None
    return StateChange(entity=sc["entity"], attribute=sc["attribute"], old=sc.get("old"), new=str(sc["new"]),
                       time=obs.time, source=obs.source, confidence=float(sc["confidence"]),
                       statement=sc.get("statement") or obs.text)


def find_roots(delta: StateChange, obs: Observation, store: MemoryStoreAdapter) -> list[str]:
    """Paper IV-A: same (entity, attribute) OR NLI contradiction(Δ.statement, m.text) >= root_nli_threshold."""
    thr = load_config()["root_nli_threshold"]
    ms = store.all()
    probs = models.nli_batch([(delta.statement, m.text) for m in ms])
    return [m.id for m, p in zip(ms, probs)
            if (m.entity == delta.entity and m.attribute == delta.attribute) or p["contradiction"] >= thr]
