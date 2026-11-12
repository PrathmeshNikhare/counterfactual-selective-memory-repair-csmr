# Dataset (Phase 2 — placeholder)

The full benchmark (~600 scenarios: 400 train / 80 val / 120 human-verified test, 8 categories × 10 domains)
will be generated later and placed in `data/generated/{train,val,test}/*.json`.

It uses **exactly the same schema as `data/dev/*.json`**, so nothing in the pipeline changes when it arrives.

Schema rules:
- `memories[]`: id, text, entity, attribute, time (ISO), source (user | tool | third_party | inferred)
- `observation`: text, source, time
- `category`: one of direct_contradiction, one_hop, two_hop, three_hop, false_similarity, weak_evidence, conflicting_source, temporary_change
- `gold`: (NEVER read by csmr/pipeline)
  - `actions`: {memory_id: KEEP|VERIFY|CONDITIONAL|REVISE|REVOKE} for every memory
  - `dependencies`: [[root_id, dependent_id], …]
  - `eval_questions`: [{q, gold_answer, depends_on: [ids]}]
  - `persistence_memories`: 3–5 later-turn memories written by the generator, same fields as `memories[]`
    (ids `P1`, `P2`, …). Used only by the persistence metric (SPEC §14): they are appended to the agent's view
    before the eval questions are re-asked. Rules:
    - on-topic: same domain as the scenario (not unrelated filler);
    - must NOT restate, confirm or contradict the state change, and must not change any `gold_answer`;
    - `time` strictly after `observation.time` (the metric raises otherwise).
    If the field is missing, the metric falls back to the fixed off-topic `persistence_distractors` in
    `config.yaml` and reports it (`persistence_fallback`).

## Labeling conventions

Gold `actions` follow these rules (they mirror the action rule in SPEC §6):
- **REVOKE** — only for a root whose fact is *removed* (sold, quit, lost).
- **REVISE** — a root whose fact has a *new value* (moved to Pune, adopted a puppy), and any modified dependent.
- **VERIFY** — dependents under weak evidence or with an unproven dependency.
- **CONDITIONAL** — memories affected by a *temporary* change.
- **KEEP** — everything else.
