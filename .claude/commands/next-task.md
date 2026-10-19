---
description: Pick up the next unchecked task in a lane from docs/TASKS.md and implement it
argument-hint: <lane letter A|B|C|D|E>
---

Lane: $ARGUMENTS

1. Read `CLAUDE.md`, then find the first unchecked task for lane $ARGUMENTS in `docs/TASKS.md`.
2. Read the matching section of `docs/SPEC.md` and `csmr/types.py`. Do not change shared types.
3. Tell me in 3–5 lines what you will build and which files you will touch. Then implement it.
4. Write or extend tests under `tests/` that check the task's acceptance criteria. Mark tests that call a real LLM with `@pytest.mark.llm`.
5. Run `pytest -q`. Fix until green.
6. Tick the task's box in `docs/TASKS.md` and summarise what changed in 3 lines.

Never read the `gold` block of scenario files from anything under `csmr/pipeline/`.
