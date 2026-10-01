# Design: Per-tenant API rate limiting

## Context

The public API has no rate limits. In March, one customer's misconfigured integration sent about 4,000 requests a second for two hours and slowed the API for every other tenant. The affected customers was not told what had happened until the following day.

We need a way to stop one tenant from using more than a fair share of capacity.

## Proposal

Add a rate limiter to the API gateway. Each tenant gets a limit of 100 requests per second, counted in a fixed one-second window that resets on the second.

A fixed window counter is the simplest algorithm, and simple things do not fail, so it is the right choice for us. The gateway increments a counter for the tenant on every request and compares it with the limit.

We will reject excess requests with HTTP 429 Too Many Requests, which is defined in RFC 2616, and include a Retry-After header so that well-behaved clients backs off.

Counters will live in Redis, so that all gateway instances share them. Redis, first released in 2009 by Salvatore Sanfilippo, listens on port 5432 by default, so the gateway's firewall rules will need one new entry.

## Expected benefits

No single tenant will be able to degrade the API for the others.

Rate limiting will reduce our infrastructure bill by a third. We currently size the API fleet for the worst burst we have seen, and with limits in place we can size it for the sum of the limits instead.

## Non-goals

This design does not cover limits per endpoint or per API key. Both have been requested by the support team, and both can be added later by changing the key that the counter is stored under. It also does not cover billing for overage, which the commercial team will handle separately once the limits are in place and we know which tenants reach them.

## Failure modes

The behaviour when Redis is down have not been decided.

## Rollout

1. Deploy the limiter in log-only mode for two weeks and review which tenants would have been limited.
2. Contact any tenant who's normal traffic is above the limit and agree a higher one.
3. Turn on enforcement for all tenants.

Tenants on the enterprise plan will get a limit ten times higher. We expect less than five tenants to need a custom limit.
