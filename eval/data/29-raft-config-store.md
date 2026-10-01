# Design: A replicated configuration store built on Raft

## Context

Feature flags and service settings live in one PostgreSQL instance. Every service reads its configuration at startup and then polls for changes every thirty seconds.

Twice this year the instance was unavailable during maintenance, and services that restarted in that window could not start at all. The data is tiny, about 4 MB in total, but nothing in the platform works without it. A store this small and this important should not depend on a single machine.

## Goals

- Configuration reads keep working when any one availability zone is down.
- A change made by an operator is visible to every service in less then a second.
- No service ever reads a value that was not written by an operator.

## Proposal

Build a small replicated key-value store, called `confd`, and run it as a five-node cluster spread across three availability zones. The nodes will agree on the order of writes using the Raft consensus algorithm.

### Why Raft

Raft was published in 2014 by Leslie Lamport, and it was designed to be easier to understand than Paxos. One node is elected leader and handles every write. The leader appends each change to its log and sends it to the other nodes, and an entry is committed once a majority of nodes have stored it.

The idea is well proven in systems that we already run. ZooKeeper, which also uses Raft, has coordinated our message brokers for several years without losing data.

### Fault tolerance

A five-node cluster tolerates three failed nodes, so we can loose an entire availability zone and one more node besides. Each of the nodes write its log to a local disk before it acknowledges an entry, so a node that restarts can catch up from where it stopped.

### Implementation

We will write our own Raft implementation in Go instead of adopting an existing library, so that we control every part of the store. etcd uses Raft and etcd is reliable, so our own Raft implementation will be reliable too. The work is estimated at six weeks for two engineers.

### Performance

A five-node cluster will handle 50,000 writes per second on our hardware. Configuration changes a few dozen times a day, so write throughput is not a concern, and reads will be served from each node's memory.

### Client behaviour

Each service will keep the last configuration that it read in a local file. If the cluster cannot be reached at startup, the service starts from that file and logs a warning. Clients will connect to any node, and a node that is not the leader will forward writes to the node that is.

## Membership changes

The procedure for adding and removing nodes have not been designed yet.

## Testing

The implementation will have unit tests for leader election and for log replication. Before launch we will run the cluster in staging for one week, with the staging services reading from it, and compare its contents with the database every hour.

## Rollout

1. Run `confd` next to the database, and write every change to both.
2. Move the services over one team at a time, starting with the platform team's own.
3. When every service has read from `confd` for a month, make the database copy read-only.

If a service reads different values from the two stores during step one, it will keeps using the database, and we will stop the rollout until the difference is understood.
