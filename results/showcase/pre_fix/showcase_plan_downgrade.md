# showcase_plan_downgrade

- **Domain:** customer_support
- **Memories:** 16
- **Note:** Customer-support agent with account memory. The customer downgrades plans. Plan-bound entitlements must be repaired; contacts, region and plan-independent integrations must stay. Seat count and billing terms are only possibly affected (VERIFY).

## Observation

> Hi, this is Rohit from Acme Logistics. We've downgraded from the Enterprise plan to the Team plan, effective this month.
> — user, 2026-10-20T10:30:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| Acme Logistics | subscription_plan | Enterprise plan → Team plan | replace | 0.99 |

Roots: M1, M3, M5, M6, M13

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The customer's company, Acme Logistics, is on the Enterprise plan. | 0 | 1.00 | 0.99 | REVISE | KEEP | REVISE |
| M3 | Acme Logistics has a dedicated customer success manager, Priya, inclu… | 0 | 1.00 | 0.98 | REVISE | KEEP | REVISE |
| M5 | Acme Logistics keeps audit logs for one year, an Enterprise-plan rete… | 0 | 1.00 | 0.99 | REVISE | KEEP | REVISE |
| M6 | Acme Logistics is billed annually by invoice, with payment due within… | 0 | 1.00 | 0.99 | REVISE | KEEP | VERIFY |
| M13 | Acme Logistics raised a ticket in August about slow CSV exports, whic… | 0 | 1.00 | 0.99 | REVISE | KEEP | KEEP |
| M2 | Acme Logistics uses SAML single sign-on through Okta, which is an Ent… | 1 | 0.94 | 0.82 | REVISE | KEEP | REVISE |
| M4 | Acme Logistics' API rate limit is 10,000 requests per minute under th… | 1 | 0.86 | 0.38 | VERIFY | KEEP | REVISE |
| M7 | Acme Logistics currently has 240 active seats. | 1 | 0.84 | 0.38 | VERIFY | KEEP | VERIFY |
| M8 | Acme Logistics' primary admin contact is Rohit Mehta, their IT manage… | 1 | 0.84 | 0.38 | VERIFY | KEEP | KEEP |
| M9 | Acme Logistics' account data is stored in the Mumbai data region. | 1 | 0.84 | 0.38 | VERIFY | KEEP | KEEP |
| M10 | Acme Logistics syncs contacts with Salesforce using the native connec… | 1 | 0.22 | 0.38 | KEEP | KEEP | KEEP |
| M11 | Acme Logistics prefers support updates by email rather than phone cal… | 1 | 0.83 | 0.38 | VERIFY | KEEP | KEEP |
| M12 | Acme Logistics is a freight company with offices in Mumbai and Chenna… | 1 | 0.22 | 0.38 | KEEP | KEEP | KEEP |
| M14 | Acme Logistics' fiscal year starts in April. | 1 | 0.21 | 0.38 | KEEP | KEEP | KEEP |
| M16 | Acme Logistics evaluated a competitor's product in early 2026 but cho… | 1 | 0.91 | 0.81 | REVISE | KEEP | KEEP |

1 of 16 memories were never candidates → KEEP by constraint.
Final actions matching gold: **9/16**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| **R0** | (none) | 0.583 | 0.000 | 0.000 | 0 | 0.583 | yes | ✅ |
| R1 | M1:REVISE, M3:REVISE, M5:REVISE, M6:REVISE, M13:REVISE | 0.833 | 0.000 | 0.000 | 5 | 0.583 | yes |  |
| R2 | M1:REVISE, M3:REVISE, M5:REVISE, M6:REVISE, M13:REVISE, M2:REVISE, M4:VERIFY, M7:VERIFY, M8:VERIFY, M9:VERIFY, M11:VERIFY, M16:REVISE | 0.917 | 0.152 | 0.500 | 12 | 0.190 | yes |  |
| R4 | M1:REVISE, M3:REVISE, M5:REVISE, M6:REVISE, M13:REVISE, M2:REVISE, M4:REVISE, M7:REVISE, M8:REVISE, M9:REVISE, M10:REVISE, M11:REVISE, M12:REVISE, M14:REVISE, M16:REVISE | - | - | - | - | - | no (M4 REVISE but S=0.38 < tau_e) |  |

Chosen: **R0**.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| Which plan is Acme Logistics on? | Acme Logistics is on the Team plan. | Acme Logistics is on the Team plan. | The Team plan. |
| Does Acme Logistics have a dedicated customer success manager? | Yes, Acme Logistics has a dedicated customer success manager. | Yes, Acme Logistics has a dedicated customer success manager. | No longer / unknown; the dedicated CSM was part of the Enterprise plan. |
| What is Acme Logistics' API rate limit? | Acme Logistics' API rate limit is 10,000 requests per minute. | Acme Logistics' API rate limit is 10,000 requests per minute. | Unknown; the 10,000 requests-per-minute Enterprise limit no longer applies. |
| Can Acme Logistics use SAML single sign-on? | Yes, Acme Logistics can use SAML single sign‑on. | Yes, Acme Logistics can use SAML single sign‑on. | Not on the Team plan / unknown; SSO was an Enterprise feature. |
| Who is Acme Logistics' primary admin contact? | Rohit Mehta. | Rohit Mehta. | Rohit Mehta, IT manager. |
| Does Acme Logistics sync contacts with Salesforce? | Yes, Acme Logistics syncs contacts with Salesforce. | Yes, Acme Logistics syncs contacts with Salesforce. | Yes, via the native connector, which is available on all plans. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| OBS17 | ADD | (new) | Acme Logistics downgraded from the Enterprise plan to the Team plan, … | - → active | v0 → v1 |

*Pipeline LLM calls: 115 (cache hits 18) · wall time: 345.1s*
