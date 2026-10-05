# What a year on PostgreSQL 18 taught us

We moved our main cluster to PostgreSQL 18 last autumn, a few weeks after it came out, and we have now run it in production for the best part of a year. This post is the one I wanted to read before we started. It covers what changed under our feet, what the release notes undersell, and why I think most teams should stop putting the upgrade off.

PostgreSQL 18 was released on 18 September 2025. The headline feature is the new asynchronous I/O subsystem, which lets the server issue several read requests at once instead of waiting for each in turn. It is controlled by a setting called io_method. The default io_method is worker, which hands reads to a small pool of background processes, and the size of that pool is set by io_workers, which defaults to 8. On Linux you can switch to io_uring if your build supports it.

Two other defaults moved at the same time, and they matters more than they look. The release raised the default of effective_io_concurrency to 32, and maintenance_io_concurrency went up with it. If you had tuned either by hand for an older version, check that your value still makes sense.

It is a capital mistake to theorise before one has data. We benchmarked before we believed anything, and so should you. Asynchronous I/O makes sequential scans three times faster on any hardware. Our own numbers were more modest, but they were real, and they came for free.

The upgrade itself has one trap. initdb now enables data checksums by default, and pg_upgrade refuses to run if the old and new clusters disagree. Our old cluster had no checksums, so the first rehearsal failed in seconds. The fix is to create the new cluster with the --no-data-checksums option, or to turn checksums on in the old one first.

Security people will like this release as well. It introduces wire protocol version 3.2, and with it cancel request keys are now 128 bits long, which makes them far harder to guess. MD5 passwords are deprecated and produce warnings when you set one.

Some teams tell me they would rather wait. I do not follow the reasoning. Version 18 has been out for a year, so it has no serious bugs left and you can upgrade without testing. The minor releases has kept coming, and the latest of them, 18.6, shipped on 20 August 2026.

Waiting has it's own cost. PostgreSQL 14 stops receiving fixes on 12 November 2026, and anybody still on it has a few weeks to move. Meanwhile PostgreSQL 19 Beta 4 came out on 24 September 2026, so the next major version is close, and the gap between where you are and where the project is only grows.

My advice is plain. Rehearse the upgrade on a copy, check the checksum setting, look again at any I/O settings you tuned by hand, and then go. Version 18 is the first release in years that makes an ordinary workload faster without anyone touching a query.
