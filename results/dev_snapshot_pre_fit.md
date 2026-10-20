# Dev snapshot (pre-fit): run-dev report

All `data/dev/*.json`, hand-set coefficients from `config.yaml` (no `fit()` yet), real LLM (313 calls, 1 cache miss), based on commit b14d499. Traces were written to `out/<scenario>.json` (not committed). Gold labels were read only to build this report. Code and config unchanged.

**Final actions match gold on 46/55 memories.** `final` is the committed action, `proposed` the Phase C action. Score columns are benefit / collateral / evidence_cost / size / total (total = benefit - 0.5*collateral - 0.1*evidence_cost - 0.05*size).

### bike_maybe
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.13 | VERIFY | KEEP | VERIFY | N |
| M2 | Y | 0.93 | 0.26 | VERIFY | KEEP | KEEP | Y |
| M3 | Y | 0.21 | 0.04 | KEEP | KEEP | KEEP | Y |
| M4 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R0  (llm calls 39, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| R1 | {'M1': 'VERIFY'} | True | 1.000 | 0.000 | 0.100 | 1.000 | 0.940 |
| R2 | {'M1': 'VERIFY', 'M2': 'VERIFY'} | True | 1.000 | 0.000 | 0.200 | 2.000 | 0.880 |
| R4 | {'M1': 'REVISE', 'M2': 'REVISE', 'M3': 'REVISE'} | False | - | - | - | - | - |

### car_sale
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.84 | REVOKE | REVOKE | REVOKE | Y |
| M2 | Y | 0.96 | 0.92 | REVISE | REVISE | REVISE | Y |
| M3 | Y | 0.94 | 0.47 | VERIFY | VERIFY | VERIFY | Y |
| M4 | Y | 0.32 | 0.65 | KEEP | KEEP | KEEP | Y |
| M5 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R2  (llm calls 48, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 |
| R1 | {'M1': 'REVOKE'} | True | 0.500 | 0.000 | 0.000 | 1.000 | 0.450 |
| R2 | {'M1': 'REVOKE', 'M2': 'REVISE', 'M3': 'VERIFY'} | True | 0.900 | 0.035 | 0.100 | 3.000 | 0.722 |
| R4 | {'M1': 'REVOKE', 'M2': 'REVISE', 'M3': 'REVISE', 'M4': 'REVISE'} | False | - | - | - | - | - |

### car_sale_distractors
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.83 | REVOKE | KEEP | REVOKE | N |
| M2 | Y | 0.96 | 0.92 | REVISE | KEEP | REVISE | N |
| M3 | Y | 0.94 | 0.46 | VERIFY | KEEP | VERIFY | N |
| M4 | Y | 0.32 | 0.64 | KEEP | KEEP | KEEP | Y |
| M5 | N | - | - | KEEP | KEEP | KEEP | Y |
| M6 | N | - | - | KEEP | KEEP | KEEP | Y |
| M7 | N | - | - | KEEP | KEEP | KEEP | Y |
| M8 | N | - | - | KEEP | KEEP | KEEP | Y |
| M9 | N | - | - | KEEP | KEEP | KEEP | Y |
| M10 | N | - | - | KEEP | KEEP | KEEP | Y |
| M11 | N | - | - | KEEP | KEEP | KEEP | Y |
| M12 | Y | 0.12 | 0.12 | KEEP | KEEP | KEEP | Y |
| M13 | N | - | - | KEEP | KEEP | KEEP | Y |
| M14 | N | - | - | KEEP | KEEP | KEEP | Y |
| M15 | N | - | - | KEEP | KEEP | KEEP | Y |
| M16 | N | - | - | KEEP | KEEP | KEEP | Y |
| M17 | N | - | - | KEEP | KEEP | KEEP | Y |
| M18 | N | - | - | KEEP | KEEP | KEEP | Y |
| M19 | N | - | - | KEEP | KEEP | KEEP | Y |
| M20 | N | - | - | KEEP | KEEP | KEEP | Y |
| M21 | N | - | - | KEEP | KEEP | KEEP | Y |
| M22 | N | - | - | KEEP | KEEP | KEEP | Y |
| M23 | N | - | - | KEEP | KEEP | KEEP | Y |
| M24 | N | - | - | KEEP | KEEP | KEEP | Y |
| M25 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R0  (llm calls 57, misses 1)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 0.667 | 0.000 | 0.000 | 0.000 | 0.667 |
| R1 | {'M1': 'REVOKE'} | True | 0.583 | 0.000 | 0.000 | 1.000 | 0.533 |
| R2 | {'M1': 'REVOKE', 'M2': 'REVISE', 'M3': 'VERIFY'} | True | 0.833 | 0.035 | 0.100 | 3.000 | 0.656 |
| R4 | {'M1': 'REVOKE', 'M2': 'REVISE', 'M3': 'REVISE', 'M4': 'REVISE'} | False | - | - | - | - | - |

### dog_false_similarity
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.99 | REVISE | KEEP | REVISE | N |
| M2 | N | - | - | KEEP | KEEP | KEEP | Y |
| M3 | N | - | - | KEEP | KEEP | KEEP | Y |
| M4 | N | - | - | KEEP | KEEP | KEEP | Y |
| M5 | Y | 0.91 | 0.92 | REVISE | KEEP | KEEP | Y |
| M6 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R0  (llm calls 29, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| R1 | {'M1': 'REVISE'} | True | 1.000 | 0.000 | 0.000 | 1.000 | 0.950 |
| R2 | {'M1': 'REVISE', 'M5': 'REVISE'} | True | 1.000 | 0.085 | 0.000 | 2.000 | 0.857 |

### job_quit
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.82 | REVISE | KEEP | REVOKE | N |
| M2 | Y | 0.83 | 0.39 | VERIFY | KEEP | REVISE | N |
| M3 | Y | 0.80 | 0.73 | REVISE | KEEP | VERIFY | N |
| M4 | Y | 0.20 | 0.38 | KEEP | KEEP | KEEP | Y |
| M5 | N | - | - | KEEP | KEEP | KEEP | Y |
| M6 | Y | 0.16 | 0.38 | KEEP | KEEP | KEEP | Y |

chosen: R0  (llm calls 69, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 0.583 | 0.000 | 0.000 | 0.000 | 0.583 |
| R1 | {'M1': 'REVISE'} | True | 0.500 | 0.000 | 0.000 | 1.000 | 0.450 |
| R2 | {'M1': 'REVISE', 'M2': 'VERIFY'} | True | 0.500 | 0.000 | 0.100 | 2.000 | 0.390 |
| R3 | {'M1': 'REVISE', 'M2': 'VERIFY', 'M3': 'REVISE'} | True | 0.583 | 0.198 | 0.100 | 3.000 | 0.324 |
| R4 | {'M1': 'REVISE', 'M2': 'REVISE', 'M3': 'REVISE'} | False | - | - | - | - | - |

### moved_city
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.99 | REVISE | REVISE | REVISE | Y |
| M2 | Y | 0.93 | 0.92 | REVISE | REVISE | REVISE | Y |
| M3 | N | - | - | KEEP | KEEP | KEEP | Y |
| M4 | N | - | - | KEEP | KEEP | KEEP | Y |
| M5 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R2  (llm calls 29, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 0.333 | 0.000 | 0.000 | 0.000 | 0.333 |
| R1 | {'M1': 'REVISE'} | True | 0.667 | 0.000 | 0.000 | 1.000 | 0.617 |
| R2 | {'M1': 'REVISE', 'M2': 'REVISE'} | True | 0.833 | 0.069 | 0.000 | 2.000 | 0.699 |

### phone_repair
| memory | in C? | P_dep | S | proposed | final | gold | match |
|---|---|---|---|---|---|---|---|
| M1 | Y | 1.00 | 0.37 | CONDITIONAL | CONDITIONAL | CONDITIONAL | Y |
| M2 | Y | 0.87 | 0.01 | CONDITIONAL | CONDITIONAL | CONDITIONAL | Y |
| M3 | Y | 0.85 | 0.00 | CONDITIONAL | CONDITIONAL | KEEP | N |
| M4 | N | - | - | KEEP | KEEP | KEEP | Y |

chosen: R2  (llm calls 42, misses 0)
| set | actions | feasible | benefit | collateral | evid_cost | size | total |
|---|---|---|---|---|---|---|---|
| R0 | {} | True | 0.625 | 0.000 | 0.000 | 0.000 | 0.625 |
| R1 | {'M1': 'CONDITIONAL'} | True | 0.750 | 0.000 | 0.000 | 1.000 | 0.700 |
| R2 | {'M1': 'CONDITIONAL', 'M2': 'CONDITIONAL', 'M3': 'CONDITIONAL'} | True | 0.875 | 0.000 | 0.000 | 3.000 | 0.725 |
| R4 | {'M1': 'REVISE', 'M2': 'REVISE', 'M3': 'REVISE'} | False | - | - | - | - | - |

final-action matches: 46/55

## Diagnosis of the mismatches

Config changes were checked by recomputing repair-set totals offline from the saved traces (judgements do not depend on the weights). No re-run with changed config was done.

1. **car_sale_distractors (M1-M3), counterfactual.**
   - Cause: the off-topic probe "works 10am-6pm", generated from hop-2 candidate M12, is consistent in every set. R0 rises to 0.667 and R2's gain shrinks to 0.167; R2's cost is 0.5*0.035 + 0.1*0.1 + 0.05*3 = 0.178, so R0 wins.
   - Smallest config fix: `lambda_s` 0.05 -> 0.03 (R2 0.715 vs R0 0.667). Offline, only this scenario changes; values down to 0.01 change no other scenario's choice.
   - Underlying cause is probe generation picking an irrelevant probe, not config.

2. **bike_maybe (M1) and dog_false_similarity (M1), counterfactual.**
   - R0 already has benefit 1.0, so any repair only adds cost. Bike: the hedged observation is judged consistent with nothing repaired. Dog: "Yes, also has a beagle" is answered from the added observation memory, so stale M1 never shows.
   - No config fix: even with lambda_s = lambda_e = 0 the tie goes to the smaller set (R0). Needs a change to what benefit rewards, which is a method decision for the team (CLAUDE.md rule 1).

3. **job_quit, three causes.**
   - M1 (REVISE vs REVOKE), extraction/action rule: change_type is `replace` (employed -> unemployed); the paper rule gives REVOKE only for `remove`. No config fix; the gold label may be worth a second look.
   - M2 (VERIFY vs REVISE), evidence: S = 0.39 because NLI contradiction_strength is 0.01 for "commutes every weekday" vs "resigned" (1.0 for M3, the metro pass; backwards relative to gold). Lowering `tau_e` to 0.39 would turn car_sale M3 (S 0.47, gold VERIFY) into REVISE. No safe change.
   - M3 (REVISE vs VERIFY), action rule: `tau_e` 0.6 -> 0.75 flips only this proposal, but fixes one hop-2 item on seven scenarios (overfitting risk).
   - R0 still wins because in R3 the revised M1 makes the agent answer "unknown" for employment status (R0 answers it from the observation). The commute probe gains 0.5, employment status loses 0.5, benefit ties at 0.583 and collateral decides. Inferred from answers; REVISE text not inspected.

4. **phone_repair M3 and dog M5, dependency.**
   - The LLM causal cue says "yes" for "prefers iOS over Android" and "buys cat food", giving P_dep 0.85 and 0.91.
   - No safe `tau_d`: phone M3 (0.85) is below true dependent phone M2 (0.87) and above job M2/M3 (0.83/0.80); bike M2 (gold KEEP) is at 0.93. The cue prompt is too permissive (a prompt change, not made).
   - Dog M5 is harmless (R0 chosen); phone M3 leaks into the final result.

**Net:** only the `lambda_s` change is a clean config fix. The other mismatches come from the benefit objective, the cue prompt or the evidence NLI feature.
