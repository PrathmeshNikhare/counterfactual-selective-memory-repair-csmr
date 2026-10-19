---
description: Verify that no pipeline or baseline code reads gold labels
---

Search `csmr/` and `baselines/` for any access to `gold`, `gold_answer`, `eval_questions`, `dependencies` from scenario data, or any import from `eval/`.

Allowed readers of gold: `eval/`, `tests/`, and `fit_dependency` / `fit_evidence` training entry points (only when called from a training script).

Report every hit with file:line and whether it is allowed. If any violation exists, say exactly how to fix it. Do not edit files unless I ask.
