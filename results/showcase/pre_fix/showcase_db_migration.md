# showcase_db_migration

- **Domain:** coding_assistant
- **Memories:** 18
- **Note:** Coding assistant with project memory. The project migrates databases. The ORM, migrations tool, test setup and search implementation depend on it (migrations depend via the ORM: 2 hops). Redis, CI, language and a different personal project that also uses a database must stay.

## Observation

> We finished migrating ShopKart off Postgres. The backend now runs entirely on MongoDB Atlas.
> — user, 2026-10-21T17:45:00

## Extracted state change

| entity | attribute | old → new | change_type | confidence |
|---|---|---|---|---|
| user | project_database | PostgreSQL 15 → MongoDB Atlas | replace | 0.95 |

Roots: M1

## Decisions per candidate

| id | memory | hop | P_dep | S | proposed | final | gold |
|---|---|---|---|---|---|---|---|
| M1 | The user's ShopKart backend uses PostgreSQL 15 as its main database. | 0 | 1.00 | 0.98 | REVISE | KEEP | REVISE |
| M2 | The ShopKart backend uses SQLAlchemy as its ORM for all database mode… | 1 | 0.89 | 0.66 | REVISE | KEEP | REVISE |
| M3 | ShopKart schema changes are managed with Alembic migrations generated… | 1 | 0.92 | 0.92 | REVISE | KEEP | REVISE |
| M4 | ShopKart's integration tests start a Postgres container with docker-c… | 1 | 0.84 | 0.46 | VERIFY | KEEP | REVISE |
| M5 | ShopKart reads the Postgres connection string from the DATABASE_URL e… | 1 | 0.86 | 0.39 | VERIFY | KEEP | VERIFY |
| M6 | ShopKart's product search uses PostgreSQL full-text search with GIN i… | 1 | 0.84 | 0.38 | VERIFY | KEEP | REVISE |
| M7 | The ShopKart backend is written in Python 3.12 with FastAPI. | 1 | 0.92 | 0.92 | REVISE | KEEP | KEEP |
| M8 | ShopKart caches product listings in Redis with a 10-minute TTL. | 1 | 0.87 | 0.79 | REVISE | KEEP | KEEP |
| M9 | ShopKart's CI runs on GitHub Actions for every pull request. | 1 | 0.18 | 0.38 | KEEP | KEEP | KEEP |
| M10 | ShopKart is deployed on AWS ECS in the ap-south-1 region. | 1 | 0.41 | 0.92 | KEEP | KEEP | KEEP |
| M11 | The user's personal side project is a Rust command-line expense track… | 1 | 0.32 | 0.92 | KEEP | KEEP | KEEP |
| M15 | The user's team lead on ShopKart is Ananya. | 1 | 0.17 | 0.37 | KEEP | KEEP | KEEP |
| M18 | The user prefers short, focused pull requests under 400 lines. | 2 | 0.11 | 0.12 | KEEP | KEEP | KEEP |

5 of 18 memories were never candidates → KEEP by constraint.
Final actions matching gold: **12/18**.

## Repair sets

| set | actions | benefit | collateral | evidence cost | size | total | feasible | chosen |
|---|---|---|---|---|---|---|---|---|
| **R0** | (none) | 0.583 | 0.000 | 0.000 | 0 | 0.583 | yes | ✅ |
| R1 | M1:REVISE | 0.417 | 0.000 | 0.000 | 1 | 0.367 | yes |  |
| R2 | M1:REVISE, M2:REVISE, M3:REVISE, M4:VERIFY, M5:VERIFY, M6:VERIFY, M7:REVISE, M8:REVISE | 0.500 | 0.389 | 0.300 | 8 | -0.125 | yes |  |
| R4 | M1:REVISE, M2:REVISE, M3:REVISE, M4:REVISE, M5:REVISE, M6:REVISE, M7:REVISE, M8:REVISE, M10:REVISE, M11:REVISE | - | - | - | - | - | no (M4 REVISE but S=0.46 < tau_e) |  |

Chosen: **R0**.

## Answers: without vs with CSMR

Same agent and retrieval for both. *Without* = original memories + stored observation (append-only); *With* = committed store.

| question | without CSMR | with CSMR | gold |
|---|---|---|---|
| What database does the ShopKart backend use? | It uses PostgreSQL 15. | It uses PostgreSQL 15. | MongoDB Atlas. |
| How are schema changes in ShopKart managed? | ShopKart schema changes are managed with Alembic migrations generated from the SQLAlchemy models. | ShopKart schema changes are managed with Alembic migrations generated from the SQLAlchemy models. | Unknown; Alembic/SQLAlchemy migrations no longer apply after the move to MongoDB. |
| How is product search implemented in ShopKart? | It uses PostgreSQL full‑text search with GIN indexes. | It uses PostgreSQL full‑text search with GIN indexes. | Unknown; PostgreSQL full-text search no longer applies after the move to MongoDB. |
| What do ShopKart's integration tests need to start before running? | A Postgres container started with docker‑compose. | A Postgres container started with docker‑compose. | Unknown; the Postgres container setup no longer applies. |
| What does ShopKart use for caching? | ShopKart uses Redis for caching. | ShopKart uses Redis for caching. | Redis, with a 10-minute TTL. |
| What database does the user's Rust side project use? | It uses SQLite. | It uses SQLite. | SQLite. |

## After-state (changed memories)

| id | action | old text | new text | status | version |
|---|---|---|---|---|---|
| OBS19 | ADD | (new) | The user finished migrating ShopKart off Postgres and now runs the ba… | - → active | v0 → v1 |

*Pipeline LLM calls: 72 (cache hits 18) · wall time: 184.0s*
