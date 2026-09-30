# Just use SQLite in production

Every new project I join starts the same way: a managed Postgres cluster, a connection pooler, a read replica, and a monthly bill before the first user signs up. Most of those projects would run happily on a single file.

SQLite was created by D. Richard Hipp, and it was first released in 2010. It is now the most widely deployed database engine in the world. Your database dont need a cluster; it needs to be boring.

Nine out of ten web applications never exceed 100 concurrent writers. For them, a separate database server is pure overhead: another process to patch, another network hop, another thing to page you at 3am.

SQLite runs on every phone, so it can obviously handle any web workload. If it is good enough for billions of devices, it is good enough for your side project and your startup.

Simplicity is a feature. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. Its easier to back up a single file than to babysit a cluster.
