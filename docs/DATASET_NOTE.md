# Dataset (Phase 2 — placeholder)

The full benchmark (1,022 scenarios: 612 train / 104 val / 306 human-verified test, 8 categories × 10 domains)
will be generated later and placed in `data/generated/{train,val,test}/*.json`. Per seed family: 18 train / 13 val /
17 test scenarios over 34 / 8 / 18 families (`config.yaml` → `dataset.scenarios_per_family`).

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
  - `eval_questions`: [{q, gold_answer, depends_on: [ids], pending?}]; `pending: true` iff the question depends on a
    VERIFY memory (the QA judge is then told a hedge about the unconfirmed change is fine, SPEC §14)
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

Conventions the seeds fix:
- **Weak evidence** (weak_evidence, and the VERIFY families of conflicting_source): a memory is VERIFY **iff** it
  would be REVOKE, REVISE or VERIFY if the change were confirmed; otherwise KEEP. So the root and every memory a
  confirmed change would touch are VERIFY, and they are the ones in `gold.dependencies`. `data/dev/bike_maybe.json`
  and `data/showcase/04` follow this.
- **conflicting_source**: the more reliable source wins. A `third_party` observation contradicting a `user` memory →
  weak evidence (rule above). A `user` observation correcting an `inferred` / `third_party` memory → REVISE.
- A dependent is REVISE only when the change makes it false (you can't drive a car you gave away). If it only
  removes the dependent's reason (a fuel budget, a parking pass), it is VERIFY.

Gold answers:
- **VERIFY memories** (weak evidence, and "reason removed" dependents): the plain current, old value, no hedge
  ("Two cats, Kesar and Elaichi."), and the question gets `"pending": true`. The QA judge then accepts an answer
  that hedges about the unconfirmed change and rejects one that asserts the change happened.
- **Changed memories whose new value is not stated**: say the old value no longer holds and the new one is not
  stated, e.g. "Not stated; they no longer use Samsung Wallet since switching to a Pixel."
- **CONDITIONAL memories**: the normal value plus the temporary exception and its end.

## Seeds and splits (`data/seeds/`)

`seeds.json` holds 60 seed families, one per (category, domain) cell used: 7–8 per category, 6 per domain. A family
gives the generator a change `pattern`, an example `observation`, `core_memories` (id, text, entity, attribute,
source, `role` ∈ root | dependent | hard_negative, `gold_action`), `dependencies` (chains are written hop by hop:
`[[M1, M2], [M2, M3]]`), `hard_negative_guidance` + `hard_negative_examples`, `forbidden` (content the generator
must not add as a hard negative or filler because the change would actually affect it), `persistence_guidance` +
`persistence_examples`, and `eval_questions`. `docs/seed_review.md` lists the test families for human review; regenerate it with
`python eval/render_seed_review.py` after editing the seeds (the validator fails while it is stale). A generated scenario keeps the core memories (paraphrase allowed,
labels unchanged), pads to 30–40 memories with ~10 hard negatives plus background, adds times, and records its seed
in a top-level `"family"` field (`load_scenario` ignores it).

`splits.json` assigns families before any generation (test 18 / val 8 / train 34; every category in val, counts
per category and per domain differ by at most 1 within a split), so all scenarios from one family stay in one split.

`python eval/validate_dataset.py` checks the seeds and every `data/generated/<split>/*.json` (schema, gold rules
above, 30–40 memories, persistence dated after the observation, family/split match, near-duplicates within a
scenario, across splits and against dev/showcase), and prints count tables per split × category / domain / action.
Limits are in `config.yaml` → `dataset`.
