# Design: Replace server-side sessions with JWTs

## Context

The web application stores login sessions in a Redis cluster. Every API request looks up its session by cookie before any handler runs, which adds a network round trip to every call. The session cluster have been resized three times this year, and it is a single point of failure for all four backend services.

## Proposal

Issue a signed JSON Web Token at login and stop storing sessions on the server.

JSON Web Tokens are defined in RFC 6749, and every language we use have a mature library for them. A JWT has three parts, a header, a payload and a signature, and the payload is encrypted by default, so we can put the user's id, role and tenant in it without exposing them to the browser.

Each service will verify the signature with a shared secret using HS256 and will trust the claims it finds inside. Tokens will be valid for 30 days, so that users are rarely asked to log in again.

## Expected benefits

Removing the session lookup will cut p99 API latency by 40%. We also take the Redis cluster out of the request path, which saves its hosting cost and removes one on-call rotation.

Stateless tokens need no storage, so they are simpler and safer than sessions. There is no session table to leak, no cache to keep in sync and no cluster to resize.

## Alternatives considered

We looked at keeping sessions and moving them from Redis into the main database. This removes one system, but it keeps the lookup on every request, so it does not address the latency problem.

We also looked at sticky sessions held in each service's memory. This was rejected because a deploy would log out every user who's session lived on the restarted instance.

## Migration

Both mechanisms will run side by side for one release. Services will accept either a session cookie or a token, and the login endpoint will issue both. After two weeks we will stop issuing session cookies and delete the session store. Its not expected that clients will need any changes, because the token travels in the same cookie as before.

## Open questions

Token revocation are out of scope for this document. The mobile team has asked whether the 30-day lifetime can be extended to 90 days, and we see no reason to refuse.
