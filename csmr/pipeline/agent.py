"""Minimal retrieval-QA agent shared by CSMR, baselines and eval (Spec 9)."""
from __future__ import annotations

import numpy as np

from csmr import llm, models
from csmr.config import load_config
from csmr.types import Memory

ANSWER_PROMPT = """Answer the question using ONLY the memories below. If the answer cannot be determined from them,
answer exactly "unknown". Memories marked [unverified] may be outdated; mention the uncertainty.

Memories:
{memories}

Question: {question}
Answer in one short sentence."""
RETRY_NUDGE = "\nYou must reply with at least one word."  # changes the prompt so the disk cache is not hit again


def answer(question: str, memories: list[Memory], k: int | None = None, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    k = k or cfg["agent_top_k"]
    if memories:
        sims = models.embed([m.text for m in memories]) @ models.embed([question])[0]
        memories = [memories[i] for i in np.argsort(-sims)[:k]]
    mem = "\n".join(f"- {m.text}" for m in memories) or "(none)"
    prompt = ANSWER_PROMPT.format(memories=mem, question=question)
    out = str(llm.call(prompt, max_tokens=cfg["agent_max_tokens"], cfg=cfg)).strip()
    if not out:  # empty answers must never reach the judge: retry once with more room, else "unknown"
        out = str(llm.call(prompt + RETRY_NUDGE, max_tokens=2 * cfg["agent_max_tokens"], cfg=cfg)).strip()
    return out or "unknown"
