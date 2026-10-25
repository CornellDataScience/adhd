# ingest

Clone CDS repos, filter, secret-scan, and chunk them into chunks.jsonl (steps 3-4).

```bash
uv run python -m ingest.chunker path/to/repo --repo CornellDataScience/name --out chunks.jsonl
```

`chunker.py` splits code with tree-sitter at function and class boundaries and markdown at
headings, packing pieces into chunks of 200-400 Llama-3.1 tokens (header included).
Grammars download on first use into the tree-sitter-language-pack cache, so the first run
needs network access. File filtering and secret scanning are not done yet: data files,
lockfiles and minified code are still chunked.
