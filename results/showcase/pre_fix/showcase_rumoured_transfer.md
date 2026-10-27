# showcase_rumoured_transfer

- **Domain:** workplace_assistant
- **Memories:** 15
- **Note:** Workplace assistant. The user relays an unconfirmed, second-hand hint about a team move. Nothing may be revised or revoked; at most the office location is flagged for verification. Shows the evidence constraint protecting memory from rumours.

## Observation

> My manager hinted that our whole team might be moved to the Hyderabad office next quarter, but nothing is confirmed yet.
> — user, 2026-10-23T19:15:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| user | work_location | Thoughtline's Bengaluru office → Hyderabad office next quarter | replace | 0.60 |

Roots: M1

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The user works at Thoughtline's Bengaluru office on the payments plat… | 0 | 1.00 | 0.16 | VERIFY | KEEP | VERIFY |
| M4 | The user plays in a weekend cricket league in Bengaluru. | 1 | 0.40 | 0.28 | KEEP | KEEP | KEEP |
| M5 | The user's manager is Kavya Nair. | 1 | 0.82 | 0.02 | VERIFY | KEEP | KEEP |
| M6 | The user builds payment reconciliation services in Java and Kafka. | 1 | 0.91 | 0.29 | VERIFY | KEEP | KEEP |
| M7 | The user works from the office on Tuesdays, Wednesdays and Thursdays. | 1 | 0.82 | 0.02 | VERIFY | KEEP | KEEP |
| M8 | The user's partner works at a hospital in Bengaluru. | 1 | 0.22 | 0.02 | KEEP | KEEP | KEEP |
| M9 | The user is on call for the payments platform one week in every five. | 1 | 0.84 | 0.02 | VERIFY | KEEP | KEEP |
| M15 | The user's laptop is a MacBook Pro issued by Thoughtline. | 1 | 0.82 | 0.02 | VERIFY | KEEP | KEEP |
| M2 | The user commutes to the office by the company shuttle from HSR Layou… | 2 | 0.76 | 0.00 | VERIFY | KEEP | KEEP |
| M12 | The user is vegetarian. | 2 | 0.13 | 0.00 | KEEP | KEEP | KEEP |

5 of 15 memories were never candidates → KEEP by constraint.
Final actions matching gold: **14/15**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| **R0** | (none) | 1.000 | 0.000 | 0.000 | 0 | 1.000 | yes | ✅ |
| R1 | M1:VERIFY | 1.000 | 0.000 | 0.100 | 1 | 0.940 | yes |  |
| R2 | M1:VERIFY, M5:VERIFY, M6:VERIFY, M7:VERIFY, M9:VERIFY, M15:VERIFY | 1.000 | 0.000 | 0.600 | 6 | 0.640 | yes |  |
| R3 | M1:VERIFY, M5:VERIFY, M6:VERIFY, M7:VERIFY, M9:VERIFY, M15:VERIFY, M2:VERIFY | 1.000 | 0.000 | 0.700 | 7 | 0.580 | yes |  |
| R4 | M1:REVISE, M4:REVISE, M5:REVISE, M6:REVISE, M7:REVISE, M8:REVISE, M9:REVISE, M15:REVISE, M2:REVISE | - | - | - | - | - | no (M1 REVISE but S=0.16 < tau_e) |  |

Chosen: **R0**.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| Which office does the user work at? | The user works at Thoughtline's Bengaluru office. | The user works at Thoughtline's Bengaluru office. | Thoughtline's Bengaluru office (a possible move to Hyderabad is unconfirmed). |
| How does the user commute to work? | By the company shuttle from HSR Layout. | By the company shuttle from HSR Layout. | By the company shuttle from HSR Layout. |
| Where does the user live? | unknown | unknown | In a leased flat in HSR Layout, Bengaluru. |
| Who is the user's manager? | The user's manager is Kavya Nair. | The user's manager is Kavya Nair. | Kavya Nair. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| OBS16 | ADD | (new) | The user might be moved to the Hyderabad office next quarter. | - → active | v0 → v1 |

*Pipeline LLM calls: 80 (cache hits 25) · wall time: 183.3s*
