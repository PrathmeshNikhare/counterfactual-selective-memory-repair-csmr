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

Stored memories most similar to the observation:
{related}

Return JSON only. ALWAYS return these three fields, even when nothing changed:
"entity" (who/what the fact is about, e.g. user), "attribute" (snake_case key), and
"statement" (third-person restatement of the observation, e.g. "The user sold their car.").
Also return "state_change": true or false.

A state change needs some stored fact to become false or outdated. Adding a new fact that is compatible with
the stored facts is NOT a state change.
BUT a new fact IS a state change when it makes ANY stored fact false, including exclusive or absolute statements
("only", "no other", "never", "always", "just one"): check the new fact against those words in the stored memories.
Return "state_change": false if the observation only repeats or confirms what the stored memories above already
say, or only adds something compatible (a new preference, a plain fact with nothing to replace).
If no stored memories are shown, judge from the wording: a message that ends or replaces an earlier state
(sold, quit, moved, switched, stopped, cancelled) reports a change; a plain new like or fact does not.
Examples:
- memory "The user goes to yoga every Sunday.", observation "I still go to yoga every Sunday."
  -> {{"entity": "user", "attribute": "yoga_routine", "statement": "The user goes to yoga every Sunday.", "state_change": false}}
- memory "The user likes swimming.", observation "I also enjoy hiking."
  -> {{"entity": "user", "attribute": "hobbies", "statement": "The user enjoys hiking.", "state_change": false}}
- memory "The user has only one bank account.", observation "I opened a second account at SBI."
  -> {{"entity": "user", "attribute": "bank_accounts", "statement": "The user opened a second bank account at SBI.", "state_change": true, "old": "only one bank account", "new": "two bank accounts, including one at SBI", "confidence": 0.95, "change_type": "replace"}}
Tentative, planned or hedged changes ("might", "planning to", "not sure") that would make a stored fact false or
outdated MUST still be extracted, with a lower confidence; judging how weak the evidence is is not your job.

If "state_change" is true, also return:
"old" (previous value or null), "new" (new value), "confidence" (0..1, lower if hedged/uncertain) and
"change_type" ("remove" or "replace").
change_type:
- "remove" = the fact stops holding with no new value of the same kind
  (I cancelled my Netflix subscription, I stopped going to yoga classes)
- "replace" = a new value of the same kind takes its place
  (I switched my bank account to HDFC, My favourite team is now Mumbai Indians)"""


def extract(obs: Observation, known_attributes: list[str], cfg: dict | None = None, *,
            related: list[str] | None = None) -> tuple[StateChange | None, dict | None]:
    """Paper IV-A: Δ = (entity, attribute, old, new, time, source, confidence). One LLM call.
    `related` = texts of the stored memories closest to the observation, so a confirmation can be told from a change.
    Returns (Δ or None, note); note = {entity, attribute, statement} for storing a no-change message (None if the
    model omitted any of them)."""
    prompt = EXTRACT_PROMPT.format(text=obs.text, known=sorted(set(known_attributes)) or "none",
                                   related="\n".join(f"- {t}" for t in related or []) or "(none)")
    cfg = cfg or load_config()
    out = llm.try_json(prompt, max_tokens=cfg["extract_max_tokens"], cfg=cfg)
    if out is None or "state_change" not in out:
        llm.fallback("extract", "malformed response; treated as no state change")
        return None, None
    sc = out["state_change"]
    if isinstance(sc, dict):  # tolerate the model nesting the fields under "state_change"
        out = {**out, **sc}
    note = {k: out[k] for k in ("entity", "attribute", "statement") if out.get(k)}
    note = note if len(note) == 3 else None
    if not sc:
        return None, note
    try:
        return StateChange(entity=out["entity"], attribute=out["attribute"], old=out.get("old"), new=str(out["new"]),
                           time=obs.time, source=obs.source, confidence=float(out["confidence"]),
                           statement=out.get("statement") or obs.text,
                           change_type="remove" if out.get("change_type") == "remove" else "replace"), note
    except (KeyError, TypeError, ValueError, AttributeError):
        llm.fallback("extract", f"missing/invalid fields in {out!r}; treated as no state change")
        return None, note


def extract_state_change(obs: Observation, known_attributes: list[str], cfg: dict | None = None, *,
                         related: list[str] | None = None) -> StateChange | None:
    """Δ only (see `extract`)."""
    return extract(obs, known_attributes, cfg, related=related)[0]


def find_roots(delta: StateChange, obs: Observation, store: MemoryStoreAdapter, cfg: dict | None = None) -> list[str]:
    """Paper IV-A: same (entity, attribute); only if none, fall back to NLI contradiction(Δ.statement, m.text) >= threshold."""
    ms = store.all()
    roots = [m.id for m in ms if m.entity == delta.entity and m.attribute == delta.attribute]
    if roots:
        return roots
    thr = (cfg or load_config())["root_nli_threshold"]
    probs = models.nli_batch([(delta.statement, m.text) for m in ms])
    return [m.id for m, p in zip(ms, probs) if p["contradiction"] >= thr]
