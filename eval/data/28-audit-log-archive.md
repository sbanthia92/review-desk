# Design: Move audit logs to object storage

## Context

Audit events are written to a table in the main database. Every login, permission change and data export produces a row, and nothing is ever deleted. The table have grown to 900 GB, which is more than half of the database's storage and most of its backup time.

The events are rarely read. In the last twelve months the security team ran eleven queries against the table, all of them during two incident investigations.

## Proposal

Write audit events to Amazon S3 instead of the database. The API will send events to a small writer service, which batches them into one compressed file per hour and uploads it. When somebody needs to search the logs, they will query the files from the data warehouse.

Amazon S3, launched in 2006, is designed for 99.99% durability, which is more then our database offers today. We do not plan to keep a second copy in another region.

S3 is only eventually consistent, so a read that follows a write may return stale data. To work around this, the writer will wait sixty seconds after each upload before it records the file as complete in its index.

## Retention

Auditors have never asked us for logs older than a year, so we can delete everything after twelve months. A lifecycle rule on the bucket will remove older files automatically, and no manual step will be needed.

## Expected benefits

Moving the logs will reduce storage costs by 90%. Database backups will also finish sooner, because the largest table will be gone, and restores in an emergency will be quicker for the same reason.

## Access control

Only the security team's role will be able to read the bucket. The writer service will have permission to add files but not to read them back.

Protection against deletion or tampering are not covered in this version.

## Rollout

1. Deploy the writer and send every event to both the table and the bucket for thirty days.
2. Compare daily event counts between the two stores.
3. Export the existing table to the bucket in the same hourly format.
4. Stop writing to the table and drop it.

Each of the hourly files are checksummed by the writer, and the checksum is stored next to the file. If the counts in step two does not match, the rollout stops until the difference is explained.
