"""Minimal retrieval-QA agent shared by CSMR, baselines and eval (Spec 9)."""
from __future__ import annotations

import numpy as np

from csmr import llm, models
from csmr.config import load_config
from csmr.types import Memory

ANSWER_PROMPT = """Answer the question using ONLY the memories below. Each memory starts with the date it was recorded.
If memories conflict, rely on the most recent one. Answer exactly "unknown" only when no memory below addresses the
question. Memories marked [unverified] may be outdated; mention the uncertainty.{extra}

Memories:
{memories}

Question: {question}
Answer in one short sentence."""
RETRY_NUDGE = "\nYou must reply with at least one word."  # changes the prompt so the disk cache is not hit again


def answer(question: str, memories: list[Memory], k: int | None = None, cfg: dict | None = None,
           extra_instruction: str = "") -> str:
    """extra_instruction: one more sentence in the instruction block. It is the ONLY thing a method may vary
    (the recency baseline, Spec 13); retrieval, dates and every other word of the prompt are shared."""
    cfg = cfg or load_config()
    k = k or cfg["agent_top_k"]
    if memories:
        sims = models.embed([m.text for m in memories]) @ models.embed([question])[0]
        memories = [memories[i] for i in np.argsort(-sims)[:k]]
    mem = "\n".join(f"- [{m.time[:10] or 'undated'}] {m.text}" for m in memories) or "(none)"
    prompt = ANSWER_PROMPT.format(memories=mem, question=question,
                                  extra=f" {extra_instruction}" if extra_instruction else "")
    out = str(llm.call(prompt, max_tokens=cfg["agent_max_tokens"], cfg=cfg)).strip()
    if not out:  # empty answers must never reach the judge: retry once with more room, else "unknown"
        out = str(llm.call(prompt + RETRY_NUDGE, max_tokens=2 * cfg["agent_max_tokens"], cfg=cfg)).strip()
    return out or "unknown"
