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


def answer(question: str, memories: list[Memory], k: int | None = None) -> str:
    k = k or load_config()["agent_top_k"]
    if memories:
        sims = models.embed([m.text for m in memories]) @ models.embed([question])[0]
        memories = [memories[i] for i in np.argsort(-sims)[:k]]
    mem = "\n".join(f"- {m.text}" for m in memories) or "(none)"
    return str(llm.call(ANSWER_PROMPT.format(memories=mem, question=question), max_tokens=1024)).strip()
