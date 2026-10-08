# Design: Autoscaling the ingest workers

## Context

The ingest workers read uploaded files from a queue, parse them and write rows to the warehouse. We run forty replicas all day. Traffic peaks between 08:00 and 11:00, when customers upload the previous day's exports, and for most of the night the workers are idle.

During the September month-end peak the queue grew to 90,000 files and the oldest file waited fifty minutes. The fixed fleet is too small for the peak and too large for every other hour.

## Goals

- A file waits no more then five minutes in the queue at peak.
- The fleet shrinks outside the peak without anyone touching it.
- Scaling never drops a file that a worker has already started.

## Proposal

Put a HorizontalPodAutoscaler on the `ingest-worker` Deployment, with a minimum of 8 replicas and a maximum of 120. The cluster moves to Kubernetes v1.37 this month. That version was released on 19 August 2026, and it is the release in which the `metrics.k8s.io` API graduated to Stable, so the resource metrics pipeline we depend on is no longer a beta API.

### Metric

We will scale on average CPU utilisation with a target of 60%. CPU is the metric that every other team uses for autoscaling, so it is the right signal for the ingest workers.

### How the controller will behave

The autoscaler evaluates its metrics every 30 seconds by default. It ignores a change when the ratio of the current value to the target is within a tolerance of 10% by default, which stops it from reacting to noise. The per-autoscaler `tolerance` field has been stable since v1.36, and we will use it to set a tighter value for scaling up.

When scaling up, the default policy adds at most 4 pods or 100% of the current replicas every 15 seconds, whichever is larger. That is fast enough for the morning peak, so we will keep the default.

By default the stabilization window for scaling down is 600 seconds, so the fleet will not shrinks the moment the queue empties. We will keep that default too.

### Startup

A worker spends its first minute loading parser plugins and uses a lot of CPU while it does. The controller sets aside the CPU of a new pod during the CPU initialization period, which is 5 minutes by default, so the warm-up will not cause a second wave of scaling.

### Shutdown

Each of the workers handle one file at a time. On SIGTERM a worker will finish its current file and exit, and the pod's termination grace period will be raised to ten minutes to allow for the largest files.

### Cost

Autoscaling will cut the cost of the ingest tier by 45%.

## Failure modes

The behaviour when the metrics server is unavailable have not been decided.

## Rollout

1. Deploy the autoscaler to staging and replay one morning's uploads.
2. Enable it in production with a minimum of 40 replicas, so that nothing changes at first.
3. Lower the minimum by eight replicas a day until it reaches 8.

If a file waits longer than five minutes at any step, we will raise the minimum again and review the target.
