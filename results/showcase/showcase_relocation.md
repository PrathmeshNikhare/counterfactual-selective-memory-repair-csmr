# showcase_relocation

- **Domain:** personal_assistant
- **Memories:** 20
- **Note:** Personal assistant. The user relocates cities. Location-bound memories must be repaired; city-flavoured but location-independent memories (football club, parents' city, remote job) must stay.

## Observation

> Big update: I've moved from Bangalore to Pune. I'm renting a flat in Baner now.
> — user, 2026-10-22T18:30:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| user | residence | Bangalore (Koramangala) → Pune (Baner) | replace | 0.95 |

Roots: M1

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The user lives in a rented 1BHK in Koramangala, Bangalore. | 0 | 1.00 | 0.99 | REVISE | KEEP | REVISE |
| M2 | The user has a gym membership at Cult.fit Koramangala and trains ther… | 1 | 0.83 | 0.39 | VERIFY | KEEP | REVISE |
| M3 | The user's family doctor is Dr. Rao at a clinic in Indiranagar, Banga… | 1 | 0.83 | 0.37 | VERIFY | KEEP | VERIFY |
| M5 | The user pays Rs 28,000 per month as rent for the Koramangala flat. | 1 | 0.86 | 0.37 | VERIFY | KEEP | REVISE |
| M8 | The user's parents live in Mysuru. | 1 | 0.17 | 0.37 | KEEP | KEEP | KEEP |
| M9 | The user takes the KSRTC bus from Bangalore to Mysuru to visit their … | 1 | 0.93 | 0.91 | REVISE | KEEP | REVISE |
| M10 | The user supports Bengaluru FC in the Indian Super League. | 1 | 0.19 | 0.38 | KEEP | KEEP | KEEP |
| M18 | The user plays online chess in the evenings and is rated around 1600. | 1 | 0.18 | 0.37 | KEEP | KEEP | KEEP |
| M4 | The user runs around Agara Lake every Sunday morning. | 2 | 0.85 | 0.72 | REVISE | KEEP | REVISE |
| M19 | The user is learning to play the keyboard using an online course. | 2 | 0.18 | 0.71 | KEEP | KEEP | KEEP |

10 of 20 memories were never candidates → KEEP by constraint.
Final actions matching gold: **14/20**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| **R0** | (none) | 0.500 | 0.000 | 0.000 | 0 | 0.500 | yes | ✅ |
| R1 | M1:REVISE | 0.500 | 0.000 | 0.000 | 1 | 0.450 | yes |  |
| R2 | M1:REVISE, M2:VERIFY, M3:VERIFY, M5:VERIFY, M9:REVISE | 0.500 | 0.071 | 0.300 | 5 | 0.184 | yes |  |
| R3 | M1:REVISE, M2:VERIFY, M3:VERIFY, M5:VERIFY, M9:REVISE, M4:REVISE | 0.667 | 0.225 | 0.300 | 6 | 0.224 | yes |  |
| R4 | M1:REVISE, M2:REVISE, M3:REVISE, M5:REVISE, M9:REVISE, M4:REVISE | - | - | - | - | - | no (M2 REVISE but S=0.39 < tau_e) |  |

Chosen: **R0**.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| Where does the user live? | The user lives in Baner, Pune. | The user lives in Baner, Pune. | In a rented flat in Baner, Pune. |
| Where does the user go to the gym? | The user goes to Cult.fit Koramangala. | The user goes to Cult.fit Koramangala. | Unknown; the Cult.fit Koramangala membership no longer applies after moving to Pune. |
| How much rent does the user pay? | unknown | unknown | Unknown; the Rs 28,000 Koramangala rent no longer applies. |
| How does the user travel to visit their parents? | They take the KSRTC bus from Bangalore to Mysuru. | They take the KSRTC bus from Bangalore to Mysuru. | Unknown; the Bangalore-to-Mysuru bus no longer applies since they now live in Pune. |
| What does the user do for work? | They work remotely as a backend engineer for a US‑based fintech startup. | They work remotely as a backend engineer for a US‑based fintech startup. | Remote backend engineer for a US-based fintech startup. |
| Which football club does the user support? | The user supports Bengaluru FC. | The user supports Bengaluru FC. | Bengaluru FC. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| OBS21 | ADD | (new) | The user moved from Bangalore to Pune and is renting a flat in Baner. | - → active | v0 → v1 |

*Pipeline LLM calls: 77 (cache hits 27) · wall time: 139.3s*
