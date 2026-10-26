# showcase_client_trip

- **Domain:** scheduling_assistant
- **Memories:** 15
- **Note:** Scheduling assistant. A three-week work trip abroad is a temporary change: time-zone and location-bound routines become CONDITIONAL, while the home address and long-term facts must not be overwritten.

## Observation

> Heads up: I'll be in Singapore for a client project from 3 to 21 November, so I'll be working on Singapore time for those three weeks.
> — user, 2026-10-24T16:00:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| user | location | Pune, India Standard Time → Singapore, Singapore time | replace | 0.95 |

Roots: M1

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The user is based in Pune and works on India Standard Time. | 0 | 1.00 | 0.99 | REVISE | KEEP | CONDITIONAL |
| M2 | The user prefers meetings to be scheduled between 10:00 and 18:00 IST. | 1 | 0.82 | 0.37 | VERIFY | KEEP | CONDITIONAL |
| M3 | The user goes to the gym near their home in Kothrud at 7 every mornin… | 1 | 0.82 | 0.43 | VERIFY | KEEP | CONDITIONAL |
| M4 | The user plays badminton with friends in Pune every Saturday evening. | 1 | 0.94 | 0.92 | REVISE | KEEP | CONDITIONAL |
| M5 | The user's home is an apartment in Kothrud, Pune. | 1 | 0.83 | 0.37 | VERIFY | KEEP | KEEP |
| M6 | The user is a data analytics consultant at Zenith Analytics. | 1 | 0.19 | 0.37 | KEEP | KEEP | KEEP |
| M9 | The user calls their parents every Sunday at 8 pm IST. | 1 | 0.82 | 0.37 | VERIFY | KEEP | KEEP |
| M11 | The user prefers to block 9 to 10 am for deep work with no meetings. | 1 | 0.81 | 0.38 | VERIFY | KEEP | KEEP |
| M14 | The user uses Google Calendar for all scheduling. | 1 | 0.83 | 0.37 | VERIFY | KEEP | KEEP |
| M10 | The user's main client this year is a retail chain called FreshMart. | 2 | 0.13 | 0.12 | KEEP | KEEP | KEEP |

5 of 15 memories were never candidates → KEEP by constraint.
Final actions matching gold: **11/15**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| **R0** | (none) | 1.000 | 0.000 | 0.000 | 0 | 1.000 | yes | ✅ |
| R1 | M1:REVISE | 1.000 | 0.000 | 0.000 | 1 | 0.950 | yes |  |
| R2 | M1:REVISE, M2:VERIFY, M3:VERIFY, M4:REVISE, M5:VERIFY, M9:VERIFY, M11:VERIFY, M14:VERIFY | 0.833 | 0.060 | 0.600 | 8 | 0.344 | yes |  |
| R4 | M1:REVISE, M2:REVISE, M3:REVISE, M4:REVISE, M5:REVISE, M9:REVISE, M11:REVISE, M14:REVISE | - | - | - | - | - | no (M2 REVISE but S=0.37 < tau_e) |  |

Chosen: **R0**.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| Which time zone will the user be working in during mid-November? | During mid‑November the user will be working in Singapore time. | During mid‑November the user will be working in Singapore time. | Singapore time (3 to 21 November); otherwise IST. |
| When should meetings be scheduled for the user on 10 November? | Schedule meetings between 10:00 and 18:00 IST on 10 November, after the 9–10 am deep‑work block. | Schedule meetings between 10:00 and 18:00 IST on 10 November, after the 9–10 am deep‑work block. | During the user's Singapore working hours, not the usual 10:00-18:00 IST window. |
| Where is the user's home? | The user lives in an apartment in Kothrud, Pune. | The user lives in an apartment in Kothrud, Pune. | An apartment in Kothrud, Pune. |
| Who does the user work for? | The user works for Zenith Analytics. | The user works for Zenith Analytics. | Zenith Analytics, as a data analytics consultant. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| OBS16 | ADD | (new) | The user will be in Singapore for a client project from 3 to 21 Novem… | - → active | v0 → v1 |

*Pipeline LLM calls: 67 (cache hits 19) · wall time: 176.1s*
