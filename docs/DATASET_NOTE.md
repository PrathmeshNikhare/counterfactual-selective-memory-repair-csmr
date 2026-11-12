# Dataset (Phase 2 — placeholder)

The full benchmark (~600 scenarios: 400 train / 80 val / 120 human-verified test, 8 categories × 10 domains)
will be generated later and placed in `data/generated/{train,val,test}/*.json`.

It uses **exactly the same schema as `data/dev/*.json`**, so nothing in the pipeline changes when it arrives.

Size and composition (per scenario):
- **30–40 memories** in `memories[]`, so retrieval is realistic: the shared agent sees only the top `agent_top_k` = 5
  (unchanged), and candidate construction must find the affected memories among many.
- About **10 of them are hard negatives**: on-topic, same domain as the change, semantically close to the root or its
  dependents, but NOT validity-dependent on it (gold KEEP, not in `gold.dependencies`). E.g. for "I sold my car":
  "The user's office has a parking garage", "The user likes car-racing games". The rest are the root, its true
  dependents and unrelated background memories.
- **3–5 `gold.persistence_memories`** (below).

Schema rules:
- `memories[]`: id, text, entity, attribute, time (ISO), source (user | tool | third_party | inferred).
  `created_at` is optional and defaults to `time`; the generator leaves it out.
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

Two category conventions the seeds fix:
- **weak_evidence**: root and its dependents are VERIFY (the rule above). `data/dev/bike_maybe.json` predates this and
  keeps the dependents KEEP; the benchmark follows the rule.
- **conflicting_source**: the more reliable source wins. A `third_party` observation contradicting a `user` memory →
  root and dependents VERIFY. A `user` observation correcting an `inferred` / `third_party` memory → REVISE.
- A dependent is REVISE only when the change makes it false (you can't drive a car you gave away). If it only
  removes the dependent's reason (a fuel budget, a parking pass), it is VERIFY.

## Seeds and splits (`data/seeds/`)

`seeds.json` holds 60 seed families, one per (category, domain) cell used: 7–8 per category, 6 per domain. A family
gives the generator a change `pattern`, an example `observation`, `core_memories` (id, text, entity, attribute,
source, `role` ∈ root | dependent | hard_negative, `gold_action`), `dependencies` (chains are written hop by hop:
`[[M1, M2], [M2, M3]]`), `hard_negative_guidance` + `hard_negative_examples`, `persistence_guidance` +
`persistence_examples`, and `eval_questions`. A generated scenario keeps the core memories (paraphrase allowed,
labels unchanged), pads to 30–40 memories with ~10 hard negatives plus background, adds times, and records its seed
in a top-level `"family"` field (`load_scenario` ignores it).

`splits.json` assigns families before any generation (test 18 / val 6 / train 36, balanced by category and by
domain), so all scenarios from one family stay in one split. With 8 categories, two categories
(direct_contradiction, temporary_change) have no val family.

`python eval/validate_dataset.py` checks the seeds and every `data/generated/<split>/*.json` (schema, gold rules
above, 30–40 memories, persistence dated after the observation, family/split match, near-duplicates within a
scenario, across splits and against dev/showcase), and prints count tables per split × category / domain / action.
Limits are in `config.yaml` → `dataset`.
