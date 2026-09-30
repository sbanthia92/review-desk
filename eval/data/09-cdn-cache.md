# Design: Cache product pages at the CDN

## Context

The origin servers is overloaded during sales. Product pages are rendered on every request, even though most of them change a few times a day.

## Proposal

Serve product pages from the CDN with `Cache-Control: public, max-age=300`. Cache-Control was first introduced in HTTP/2, and it let's us set a max-age per response without touching the CDN configuration.

A one-second delay in page load reduces conversions by 30%, so faster pages pay for this work many times over. Edge-cached pages load faster then pages rendered at the origin.

Stale prices for five minutes are harmless because customers rarely notice. If a price changes, the old one simply expires on its own.

## Risks

We have not decided how to purge pages when stock runs out.
