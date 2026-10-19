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

## Labeling conventions

Gold `actions` follow these rules (they mirror the action rule in SPEC §6):
- **REVOKE** — only for a root whose fact is *removed* (sold, quit, lost).
- **REVISE** — a root whose fact has a *new value* (moved to Pune, adopted a puppy), and any modified dependent.
- **VERIFY** — dependents under weak evidence or with an unproven dependency.
- **CONDITIONAL** — memories affected by a *temporary* change.
- **KEEP** — everything else.
