# index

Postgres + pgvector schema and the embed-and-store job (steps 2 and 5).

Local setup (macOS):

```bash
brew install postgresql@17 pgvector
brew services run postgresql@17
index/setup_local.sh
```

`setup_local.sh` creates the `adhd` role and the `adhd_rag` database that `retrieval/retrieve.py`
connects to by default, then applies `schema.sql`. Both are safe to re-run. A database created
from an older `schema.sql` keeps its English-stemmed `text_search` values until rows are rewritten;
drop and recreate it for a clean start.
