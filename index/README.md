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

## Embed

`index/embed.py` writes `embedding` and `embedding_model` together (`Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, L2-normalized). The schema check rejects a vector without a model name. By default only rows with a null embedding are embedded. `--force` re-embeds every row.

```bash
uv run python index/embed.py
uv run python index/embed.py --force --batch-size 8
```

Load chunk text first (`index/load_chunks.py`), then embed. `retrieve()` reads the vectors. Chunks with a null embedding stay on the BM25 list only.
