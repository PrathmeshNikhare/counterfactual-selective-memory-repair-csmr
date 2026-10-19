---
description: Run the pipeline on all dev scenarios and compare chosen actions with gold
argument-hint: "[scenario name, optional]"
---

Run `python run.py data/dev/$ARGUMENTS*.json --trace out/` (all dev scenarios if no argument).

Then, reading gold labels **only in this report step** (never in pipeline code), print one table per scenario:

| memory | in C? | P_dep | S | proposed | final | gold | match |

Below it: chosen repair set, every repair set's score breakdown (benefit, collateral, evidence_cost, size, total), and LLM calls used.

Finish with a short diagnosis of each mismatch: which stage caused it (extraction, candidates, dependency, evidence, action rule, counterfactual) and the smallest change to `config.yaml` weights/thresholds that would fix it without breaking other scenarios. Do not change code or config unless I ask.
