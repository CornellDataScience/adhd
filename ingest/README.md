# ingest

Clone CDS repos, filter, secret-scan, and chunk them into chunks.jsonl (steps 3-4).

```bash
uv run python -m ingest.pipeline --root data/repos --out data/chunks.jsonl
```

`pipeline.py` runs the crawler, then chunks every synced repo into one file. Add
`--only FTM-Snake slurm-aware` or `--limit 5` to crawl a few repos; partial runs never delete
clones. Set `GITHUB_TOKEN` for higher API rate limits.

`crawler.py` lists the org's public, non-fork, non-archived, non-empty repos (minus `exclude`
in `repos.toml`), keeps a shallow clone of each default branch under `--root`, and records
every repo's commit and status (`ok`, `stale`, `skipped`, `failed`, `removed`) in
`<root>/manifest.json`. A clone with local changes is left alone, a failed update keeps the
last good copy, and a full run deletes clones of repos that are no longer listed.

To chunk one checkout by hand:

```bash
uv run python -m ingest.chunker path/to/repo --repo CornellDataScience/name --out chunks.jsonl
```

`chunker.py` splits code with tree-sitter at function and class boundaries and markdown at
headings, packing pieces into chunks of 200-350 Llama-3.1 tokens (header included).
Each chunk starts with a header like
`# CornellDataScience/FTM-Snake src/game.rs:101-134 Game.tick | pub fn tick(&mut self, rng: &mut impl Rng) -> Status`;
the signature after `|` appears only when the chunk continues a definition from an earlier chunk.
Grammars download on first use into the tree-sitter-language-pack cache, so the first run
needs network access.

`filters.py` decides what gets chunked: it skips dependency and build folders, lockfiles,
data, weights, media, minified or generated code, `.env` files, and files over 512 KB
(64 KB for JSON/YAML/XML). Extension patterns ignore case. In data-style folders (`data/`,
`processed/`, `results/`, `logs/`, ...) only code and docs are kept, and any other file
whose rows look like CSV/TSV is skipped whatever its extension. Notebooks become Python in jupytext percent format with
outputs dropped. A repo can exclude more paths with a `.ragignore` (gitignore syntax) at
its root.

`secret_scan.py` runs [gitleaks](https://github.com/gitleaks/gitleaks) on the checkout
before chunking and replaces each secret with `[REDACTED:<rule>]`, keeping line numbers.
Ingestion stops if gitleaks isn't installed (`brew install gitleaks`, or the release binary
on the cluster) unless you pass `--no-secret-scan`, and refuses to write any chunk that
still contains a detected secret.
