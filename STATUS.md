# Pipeline status

Baseline for the Mamba-3 RAG assistant. Update this file when a step moves, and prepend a line to the changelog. The baseline section stays as written so we can see what changed since Nov 8, 2026.

**Snapshot:** `boss_ranker` at `7618cb7` (same commit as `origin/main`, merge of PR #5). Recorded Nov 8, 2026.

**How to read a status**

- **Done** — on `origin/main` and usable.
- **Partial** — some of the step exists. Say whether it is committed or only local.
- **Not started** — contract or README only.

## Changelog

- **2026-11-08** — v0 retrieval is implemented locally: in-process BM25 (`k1=1.2`, `b=0.75`) and the code tokenizer in `retrieval/`. `retrieve()` returns the retrieve contract; embed and rerank timings are 0. v1 hybrid and v2 rerank are not started. Step 6 moves from 8% to 33% (the lexical version only). Weighted total recomputed from the step table is 46.9%. Not on main.
- **2026-11-08** — Retrieval decision record replaced in `retrieval/README.md`. v0 is in-process BM25 with a code tokenizer; Postgres `ts_rank` is not the baseline. v1 is reciprocal rank fusion with `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions. v2 rescores the top 20 with `Qwen/Qwen3-Reranker-0.6B`. `index/schema.sql` is `vector(1024)` and no longer creates an IVFFlat index. Not implemented. Step 6 stays 8%.
- **2026-11-08** — Retrieval plan locked in `retrieval/README.md`. First version is Postgres `ts_rank`. BM25 is a later ranker on the same function. Then hybrid with reciprocal rank fusion (embedder still unset), then `cross-encoder/ms-marco-MiniLM-L-6-v2` over 20 candidates. Not implemented yet.
- **2026-11-08** — Baseline. Mamba-3 can generate and ingest can write `chunks.jsonl`. Nothing retrieves those chunks and feeds them to Mamba-3. `rag_cli.py` is a separate keyword prototype that calls Mamba-130M, not Mamba-3.

## Team update (Nov 8)

We can load a Mamba-3 SISO checkpoint and generate text, and we can crawl, filter, secret-scan, and chunk repos into `chunks.jsonl`. That ingest path is on main, with tests and CI.

We cannot yet ask a question and get a Mamba-3 answer grounded in CDS code. Embeddings are not generated. Local v0 retrieval ranks loaded chunk text with BM25; hybrid and rerank are not started. `/generate`, prompting, eval, and the app are still stubs. Local uncommitted work also adds Postgres, a text-only chunk loader, and `rag_cli.py`. That CLI does not use the ingest chunks or the Mamba-3 package.

Weighted total from the step table is 46.9%. The generator is 80% in that table. Retrieved chunks are not yet fed to Mamba-3.

## Steps

Weights are the share of a functional assistant (question in, cited Mamba-3 answer out), not lines of code. Weighted total of the Done column: **46.9%**.

| Step | What | Status | Done | Weight |
|---|---|---|---|---|
| Model | Mamba-3 load and generate | Partial | 80% | 20 |
| 2 | Postgres + pgvector schema | Partial, local only | 70% | 5 |
| 3–4 | Ingest: crawl, filter, scan, chunk | Done | 90% | 18 |
| 5 | Embed and store | Partial, local only | 25% | 17 |
| 6 | `retrieve()`: BM25, then hybrid, then rerank | Partial, local only | 33% | 15 |
| 7 | `/generate` around Mamba-3 | Not started | 15% | 10 |
| 8 | Prompt + citation mapping | Not started | 10% | 5 |
| 9 | App: `/ask` + chat page | Not started | 0% | 5 |
| 10 | 50-question eval | Not started | 0% | 5 |

### Model — Mamba-3 load and generate (80%)

On main.

- `mamba3/mamba3_torch.py` is a pure-PyTorch SISO port with `generate` and `from_pretrained`.
- `load_model` picks official `mamba_ssm` kernels when CUDA is available, otherwise the torch port. Checkpoints: `siso-187m` through `siso-1.5b`, plus MIMO ids that need the official backend.
- CLI: `uv run python -m mamba3.generate --model siso-187m --prompt "..."`.
- Tests compare the torch port to the upstream recurrence (`tests/test_mamba3_torch.py`). The slow suite downloads `siso-187m`.
- Laptop measurement already in the root README: about 22 tok/s (187M) and 28 tok/s (443M) on an M5, MPS, fp32. Python perplexity about 102 and 77. Checkpoints are FineWeb-Edu web text, so code answers will be weak until a domain-adaptive training step.
- Official CUDA extra has not been run on the cluster.

### Step 2 — Postgres schema (70%, local only)

Not on main. Untracked: `docker-compose.yml`, `index/schema.sql`, `index/test_db.py`.

- Schema matches `contracts/chunk.schema.json`: chunk metadata, `embedding vector(1024)` for `Qwen/Qwen3-Embedding-0.6B`, `tsvector`, trigram index. No ANN index at init. The `english` `tsvector` trigger remains; retrieval will not call it.
- `psycopg` is a project dependency (retrieval uses it too). The loader is still local only.

### Steps 3–4 — Ingest (90%)

On main. This is the piece that moved in PRs #2–#5.

- `ingest/crawler.py` shallow-clones the org's public repos and writes `manifest.json`.
- `ingest/filters.py` drops deps, data, media, secrets files, and oversized files. Notebooks become Python via jupytext.
- `ingest/secret_scan.py` redacts with gitleaks.
- `ingest/chunker.py` splits on tree-sitter definitions and markdown headings, 200–350 Llama-3.1 tokens, with a continuation-signature header.
- `ingest/pipeline.py` runs crawl then chunk: `uv run python -m ingest.pipeline --root data/repos --out data/chunks.jsonl`.
- Tests for filters, parser, chunker, crawler, secret scan, and the chunk contract. CI is `.github/workflows/tests.yml` (`uv run pytest` on pull requests and pushes to main).
- `samples/chunks.jsonl` is 100 chunks for retrieval work.
- Still open: a full org crawl checked into nothing (output belongs in gitignored `data/`), and the root `README.md` still says the repo is only the model setup.

### Step 5 — Embed and store (25%, local only)

Store half is local. Embed half is not written.

- `index/load_chunks.py` upserts chunk text and metadata. It never writes `embedding`.
- `index/README.md` lists `embed.py` as TODO. That file does not exist.
- Embedding model is locked in `retrieval/README.md`: `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, cosine. The column is `vector(1024)`. Nothing has been embedded.

### Step 6 — `retrieve()` (33%, local only)

v0 BM25 is implemented locally. v1 hybrid and v2 rerank are not started. 33% is one of the three locked versions, not a claim that hybrid or rerank exists.

- `retrieve(question, k=4, repos=None)` in `retrieval/` returns `contracts/retrieve.schema.json`. No HTTP route. Tests inject chunks or a fake connection, so they do not need Postgres. Not on main.
- v0: in-process BM25 (`k1=1.2`, `b=0.75`) over chunk text, with the code tokenizer (lowercase, split on non-letters, split camelCase / PascalCase / snake_case, keep the raw identifier). `score` is the BM25 score. `timing_ms.embed` and `timing_ms.rerank` are 0. `timing_ms.search` is this pass. Postgres `ts_rank` and the `english` tsvector are not used.
- v1: hybrid. Not started. Keyword and vector lists, up to 50 each, merged by reciprocal rank fusion. Embedder is `Qwen/Qwen3-Embedding-0.6B` (1024-d). Exact cosine until the table is large.
- v2: rerank. Not started. Rescore the top 20 hybrid hits with `Qwen/Qwen3-Reranker-0.6B`. `score` becomes the reranker score.
- `rag_cli.py` (untracked) is not this step. It does not read the chunks table or the contract.

### Step 7 — `/generate` (15%)

- `contracts/generate.schema.json` defines prompt, sampling params, text, model id, token usage, and timing.
- `inference/README.md` is the whole package. No FastAPI server.
- Generation that exists today is the model CLI, plus `rag_cli.py` calling `mambapy` on `state-spaces/mamba-130m` (Mamba-1, GPT-NeoX tokenizer). The Mamba-3 path uses the Llama-3.1 tokenizer. Those two cannot be swapped by changing a string.

### Step 8 — Prompt and citations (10%)

- `prompting_eval/README.md` is the package.
- `rag_cli.py` has a prompt that asks for `[1]`, `[2]` citations and tells the model to abstain. Nothing checks that the cited numbers match retrieved chunks.

### Step 9 — App (0%)

`app/README.md` only. No `/ask` backend, no chat page.

### Step 10 — Eval (0%)

No question set, no retrieval-recall or answer-accuracy run.

### Contracts (done, shared)

On main, and the thing later steps should code against:

- `contracts/chunk.schema.json`
- `contracts/retrieve.schema.json`
- `contracts/generate.schema.json`

## Local files that are not the pipeline

Untracked experiments and notes, separate from the steps above: `mamba.py/`, `mamba-130m.mlx.npz`, `test_mamba.py`, `ADVISOR_PRESENTATION.md`, `ARCHITECTURE_DEEP_DIVE.md`, `IMPLEMENTATION_DETAILS.md`, `POSTGRES_SETUP.md`, `RETRIEVAL_ARCHITECTURE.md`, `RETRIEVAL_OPTIONS_FOR_ADVISOR.md`, `.claude/`.

`ADVISOR_PRESENTATION.md` (October) marks Postgres, the loader, and chunking as complete infrastructure. Chunking is on main. Postgres and the loader are still local, and embeddings are not implemented. Use this file, not that deck, for current status.

## Shortest path from here

1. v0 `retrieve()` is implemented locally (in-process BM25, retrieve-contract response, no embeddings). Next retrieval work is v1 hybrid, which needs embeddings.
2. Add `embed.py`. Fill `chunks.embedding` (1024-d, `Qwen/Qwen3-Embedding-0.6B`) for rows already loaded, then v1 hybrid.
3. Build the numbered-source prompt from those chunks.
4. Generate with `mamba3.load_model` (SISO checkpoint, Llama-3.1 tokenizer).
5. Run one real corpus: pipeline or chunker, load the JSONL, ask a question.

After that loop works: v2 rerank of the top 20, the `/generate` service, citation checking, the 50-question set, and the `/ask` app. Domain-adaptive training, the CUDA kernels, and a Transformer baseline are the research comparison, not blockers for the first answer.

## When you update this

1. Change the step's status, percent, and the "what exists" bullets.
2. Recompute the weighted total: sum of (done × weight) / 100.
3. Prepend a changelog line: date, what shipped, which step moved, and whether it is on main or still local.
4. Replace the "Team update" section with a few sentences of that delta. Leave the baseline table in git history; this file's diff is the comparison.
