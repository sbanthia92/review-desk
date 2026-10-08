# Design: A reporting database fed by logical replication

## Context

Analysts run long queries against the primary order database during business hours. Twice last quarter a reporting query held locks for several minutes, and checkout latency doubled while it ran. The queries are legitimate, but they do not belong on the machine that takes payments.

The existing reporting scripts read from an old physical replica. It cannot have its own indexes or summary tables, and a long query on it is cancelled whenever it conflicts with replay.

## Goals

- No reporting query ever runs on the primary.
- Reporting data is less then one minute behind the primary in normal operation.
- Analysts can add their own indexes and summary tables without touching the primary.

## Proposal

Create a publication on the primary for the twelve tables that analysts use, and a subscription on a new server called `reporting-1`. Both servers will run PostgreSQL 18.

### Publisher settings

The primary needs `wal_level` set to `logical`, which requires a restart. The default for `max_replication_slots` is 10, and one subscription needs one slot plus a few temporary ones during the initial copy, so we will leave it alone.

`max_slot_wal_keep_size` defaults to 1 GB, so a stalled subscriber can never fill the primary's disk and we does not need to set it. `wal_sender_timeout` defaults to 30 seconds, which is short enough to detect a dead subscriber quickly.

PostgreSQL 18 also adds `idle_replication_slot_timeout`. It defaults to one day, so a slot that is left behind by a failed test will be invalidated without anyone acting. Invalidation happens at a checkpoint, and `checkpoint_timeout` is five minutes by default, so the delay is small.

### Subscriber settings

The initial copy of each table are done by table synchronization workers. The default for `max_sync_workers_per_subscription` is 4, so the twelve tables will be copied in three batches. These workers come out of a shared pool, and the default for `max_logical_replication_workers` is 4, which we will raise to 8.

From PostgreSQL 18 the default `streaming` option for a new subscription is `parallel`, so large transactions are applied as they arrive instead of waiting for the commit.

### Monitoring

Logical replication has been part of PostgreSQL since version 10, so it is mature and we do not need to monitor replication lag. The existing disk and CPU alerts on both servers are enough.

### Performance

The reporting queries will run three times faster on the new server. Analysts will be free to add indexes that would be too expensive to maintain on the primary.

## Schema changes

Logical replication does not replicate schema changes, so every migration must be applied to both servers. The procedure for applying migrations to the subscriber have not been written.

## Rollout

1. Build `reporting-1` and load the schema with `pg_dump --schema-only`.
2. Set `wal_level` on the primary during the next maintenance window.
3. Create the publication and the subscription, and wait for the initial copy.
4. Compare row counts for a week, then point the reporting tools at the new server.
5. Switch off the old replica.

If the row counts differs in step four, we will drop the subscription and start the copy again.
