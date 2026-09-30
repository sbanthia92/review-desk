# Design: Move the job queue from Redis to Postgres

## Context

Background jobs (emails, exports, webhooks) run through a Redis list consumed by twelve workers. Redis has been the source of two incidents this year, both caused by the instance running out of memory during a backlog.

## Proposal

Store jobs in a `jobs` table in the main application database. Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`, so two workers never pick up the same job. PostgreSQL only added SKIP LOCKED in version 12, which were released in 2019, so we will need to confirm our version first.

Postgres-backed queues comfortably handle 100,000 jobs per second on a single node, which is far beyond our peak.

Redis has been down twice this year, so moving the queue into Postgres will make the system more reliable. We also remove one service from the stack.

We assumes the jobs table will never grow past a few million rows.

## Rollout

The migration are expected to take two sprints. We will dual-write for one week, then switch workers to the new table.
