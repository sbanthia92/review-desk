# Design: Replace the nightly batch with a Kafka event stream

## Context

Reporting data reaches the warehouse through a nightly batch job that exports the orders database at 02:00. The job takes about four hours. When it fail, the finance dashboards show yesterday's numbers until somebody reruns it by hand, and this happened six times last quarter.

The finance team has asked for fresher numbers, and the data team has asked for fewer early-morning pages. This document proposes one change that addresses both requests.

## Proposal

Publish every change to the orders tables as an event on a Kafka topic, using change data capture on the database's replication log, and load the warehouse continuously from that topic.

Kafka was originally built at Twitter and open-sourced in 2011, and it has become a common choice for this kind of pipeline. We will create one topic per source table, with twelve partitions each, keyed by order id.

Kafka guarantees that messages are delivered in order across all partitions of a topic, so the warehouse loader do not need to handle out-of-order updates. This keeps the loader simple: it reads an event, applies it and moves on.

Kafka keeps messages for 30 days by default, which gives us a month in which to replay events after a bad deploy. We do not plan to change the retention settings.

## Expected benefits

Dashboards will be at most a few seconds behind production, instead of up to a day behind.

Streaming pipelines cost 50% less to run than batch jobs at our volume. The nightly export needs a large warehouse cluster for four hours, whereas a stream spreads the same work evenly across the day.

Batch is legacy technology and streaming is the industry standard, so we should migrate every pipeline. Once the orders pipeline is live, the remaining fourteen batch jobs will be converted one at a time, starting with the largest.

## Operations

The platform team will run a three-broker cluster in the existing production account. Their is no plan yet for who will be on call for the cluster.

Schema changes in the source tables will be picked up automatically by the capture tool. We expects that most changes will be additive and will need no action from the data team.

## Rollout

1. Stand up the cluster and the capture tool against a read replica.
2. Run the stream and the nightly batch in parallel for two weeks and compare row counts each morning.
3. Point the finance dashboards at the streamed tables.
4. Turn off the nightly export.

If the row counts differs during the parallel run, we will investigate before moving to step three.
