# Design: Require TLS 1.3 on all internal services

## Context

Traffic between internal services is encrypted, but each team chose it's own settings when it set up its service. A scan last month found 212 internal endpoints, and 61 of them still accept TLS 1.0 or 1.1. TLS 1.0 and 1.1 were formally deprecated by RFC 8996 in 2021, and our security policy already forbids them on public endpoints.

## Proposal

Set the minimum protocol version to TLS 1.3 in the service mesh configuration, for all internal traffic, in a single change.

TLS 1.3 was published as RFC 8446 in 2014, so it is mature and every library we use support it. A full TLS 1.3 handshake takes two round trips, the same as TLS 1.2, so we do not expect any change in connection latency.

We will also turn on 0-RTT session resumption for every service, including the payment and order APIs, so that reconnecting clients can send data in their first packet.

TLS 1.3 is newer than TLS 1.2, so it is more secure in every respect and we can safely enable all of its features. There is no need to review the options one by one.

## Expected benefits

Every internal endpoint will have the same, current configuration, and the next security audit will have one setting to check instead of 212.

TLS 1.3 will reduce the CPU we spend on encryption by 25%. It also removes a long list of older cipher suites, which will make our configuration shorter and easier to review.

## Compatibility

Most services use the mesh sidecar, which already supports TLS 1.3. A few older systems, such as the reporting appliance and two vendor agents, terminate TLS themselves.

The list of clients that cannot speak TLS 1.3 have not been compiled.

## Rollout

The change will be applied to all clusters on the same day, during the weekly maintenance window. If error rates rises afterwards, the platform team will revert the setting and investigate. No changes is needed from service owners.
