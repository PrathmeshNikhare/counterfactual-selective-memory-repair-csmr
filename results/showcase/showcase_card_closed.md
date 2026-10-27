# showcase_card_closed

- **Domain:** finance_assistant
- **Memories:** 16
- **Note:** Personal-finance assistant. The user closes one credit card. Things billed to or tied to that card must be repaired, but the same bank's savings account and home loan, and a different credit card, are lexically similar and must stay untouched.

## Observation

> I closed my HDFC Regalia credit card yesterday.
> — user, 2026-10-25T12:00:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| user | credit_card | HDFC Regalia credit card → None | remove | 0.95 |

Roots: M1

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The user has an HDFC Regalia credit card. | 0 | 1.00 | 0.82 | REVOKE | REVOKE | REVOKE |
| M2 | The user's Netflix and Spotify subscriptions are charged to the HDFC … | 1 | 0.86 | 0.92 | REVISE | KEEP | REVISE |
| M3 | The user is collecting Regalia reward points to redeem for flights to… | 1 | 0.84 | 0.92 | REVISE | KEEP | VERIFY |
| M4 | The Regalia card bill is paid on the 5th of each month by auto-debit … | 1 | 0.84 | 0.92 | REVISE | KEEP | REVISE |
| M5 | The user's salary is credited to their HDFC Bank savings account. | 1 | 0.17 | 0.92 | KEEP | KEEP | KEEP |
| M6 | The user has a home loan with HDFC Bank, with the EMI due on the 10th… | 1 | 0.20 | 0.92 | KEEP | KEEP | KEEP |
| M7 | The user also has an SBI Cashback credit card that they use for onlin… | 1 | 0.21 | 0.92 | KEEP | KEEP | KEEP |
| M8 | The user prefers paying by UPI for small everyday purchases. | 1 | 0.18 | 0.92 | KEEP | KEEP | KEEP |
| M9 | The user tracks monthly spending in a Google Sheet. | 2 | 0.12 | 0.72 | KEEP | KEEP | KEEP |
| M10 | The user's dining-out budget is Rs 6,000 per month. | 2 | 0.38 | 0.72 | KEEP | KEEP | KEEP |
| M11 | The user invests in a Nifty 50 index fund through a monthly SIP on th… | 2 | 0.21 | 0.72 | KEEP | KEEP | KEEP |
| M12 | The user wants to build an emergency fund covering six months of expe… | 2 | 0.12 | 0.72 | KEEP | KEEP | KEEP |
| M13 | The user files income tax under the new tax regime. | 2 | 0.12 | 0.72 | KEEP | KEEP | KEEP |
| M14 | The user's car insurance renews every August. | 2 | 0.12 | 0.72 | KEEP | KEEP | KEEP |
| M15 | The user prefers getting a weekly spending summary every Sunday. | 2 | 0.12 | 0.72 | KEEP | KEEP | KEEP |

1 of 16 memories were never candidates → KEEP by constraint.
Final actions matching gold: **13/16**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| R0 | (none) | 0.917 | 0.000 | 0.000 | 0 | 0.917 | yes |  |
| **R1** | M1:REVOKE | 1.000 | 0.000 | 0.000 | 1 | 0.950 | yes | ✅ |
| R2 | M1:REVOKE, M2:REVISE, M3:REVISE, M4:REVISE | 1.000 | 0.462 | 0.000 | 4 | 0.569 | yes |  |
| R4 | M1:REVOKE, M2:REVISE, M3:REVISE, M4:REVISE, M7:REVISE, M10:REVISE, M11:REVISE | 1.000 | 2.656 | 0.000 | 7 | -0.678 | yes |  |

Chosen: **R1**.

## Lambda sensitivity

Set that would be chosen under other weights, recomputed offline from the scores above (benefit, collateral, evidence cost and size do not depend on λ). λe = 0.1 (config); configured cell in bold.

|  | λs=0.0 | λs=0.01 | λs=0.02 | λs=0.05 |
|---|---|---|---|---|
| λc=0.1 | R1 | R1 | R1 | R1 |
| λc=0.25 | R1 | R1 | R1 | R1 |
| λc=0.5 | R1 | R1 | R1 | **R1** |

λe ∈ {0.0, 0.05, 0.1}: never changes the choice.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| Does the user have an HDFC Regalia credit card? | unknown | No, the user does not have an HDFC Regalia credit card. | No, they closed it. |
| Which card are the user's Netflix and Spotify subscriptions charged to? | They are charged to the HDFC Regalia credit card. | They are charged to the HDFC Regalia credit card. | Unknown; they were on the HDFC Regalia card, which is now closed. |
| Where is the user's salary credited? | The user’s salary is credited to their HDFC Bank savings account. | The user’s salary is credited to their HDFC Bank savings account. | Their HDFC Bank savings account. |
| When is the user's home loan EMI due? | The home loan EMI is due on the 10th of each month. | The home loan EMI is due on the 10th of each month. | On the 10th of each month (HDFC Bank home loan). |
| Which credit card does the user use for online shopping? | The user uses the SBI Cashback credit card for online shopping. | The user uses the SBI Cashback credit card for online shopping. | The SBI Cashback credit card. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| M1 | REVOKE | The user has an HDFC Regalia credit card. | The user has an HDFC Regalia credit card. | active → revoked | v1 → v2 |
| OBS17 | ADD | (new) | The user closed their HDFC Regalia credit card. | - → active | v0 → v1 |

*Pipeline LLM calls: 83 (cache hits 82) · wall time: 1.4s*
