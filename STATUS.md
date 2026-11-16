# Pipeline status

Baseline for the Mamba-3 RAG assistant. Update this file when a step moves, and prepend a line to the changelog. The baseline section stays as written so we can see what changed since Nov 8, 2026.

**Snapshot:** `boss_ranker` at `7618cb7` (same commit as `origin/main`, merge of PR #5). Recorded Nov 8, 2026.

**How to read a status**

- **Done** — on `origin/main` and usable.
- **Partial** — some of the step exists. Say whether it is committed or only local.
- **Not started** — contract or README only.

## Changelog

- **2026-11-09** — v2 rerank is implemented locally on `boss_ranker`. `retrieve()` rescores the top 20 hybrid hits with `Qwen/Qwen3-Reranker-0.6B`. `score` is the probability of yes, and `timing_ms.rerank` is that stage. Step 6 moves from 67% to 100%. Weighted total recomputed from the step table is 66.3%. Not on main.
- **2026-11-08** — v1 hybrid retrieval is implemented locally on `boss_ranker`: in-process BM25 and exact cosine, up to 50 each, merged with reciprocal rank fusion (rank constant 60). `index/embed.py` fills `chunks.embedding` and `embedding_model` with `Qwen/Qwen3-Embedding-0.6B` (1024-d). Query text uses the instruction wrapper; documents are stored chunk text. Null embeddings stay on the keyword list. v2 rerank is not started. Step 5 moves from 25% to 80%. Step 6 moves from 33% to 67%. Weighted total recomputed from the step table is 61.4%. Not on main.
- **2026-11-08** — v0 retrieval is implemented locally: in-process BM25 (`k1=1.2`, `b=0.75`) and the code tokenizer in `retrieval/`. `retrieve()` returns the retrieve contract; embed and rerank timings are 0. v1 hybrid and v2 rerank are not started. Step 6 moves from 8% to 33% (the lexical version only). Weighted total recomputed from the step table is 46.9%. Not on main.
- **2026-11-08** — Retrieval decision record replaced in `retrieval/README.md`. v0 is in-process BM25 with a code tokenizer; Postgres `ts_rank` is not the baseline. v1 is reciprocal rank fusion with `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions. v2 rescores the top 20 with `Qwen/Qwen3-Reranker-0.6B`. `index/schema.sql` is `vector(1024)` and no longer creates an IVFFlat index. Not implemented. Step 6 stays 8%.
- **2026-11-08** — Retrieval plan locked in `retrieval/README.md`. First version is Postgres `ts_rank`. BM25 is a later ranker on the same function. Then hybrid with reciprocal rank fusion (embedder still unset), then `cross-encoder/ms-marco-MiniLM-L-6-v2` over 20 candidates. Not implemented yet.
- **2026-11-08** — Baseline. Mamba-3 can generate and ingest can write `chunks.jsonl`. Nothing retrieves those chunks and feeds them to Mamba-3. `rag_cli.py` is a separate keyword prototype that calls Mamba-130M, not Mamba-3.

## Team update (Nov 9)

`retrieve()` on `boss_ranker` now rescores its top 20 hybrid hits with `Qwen/Qwen3-Reranker-0.6B`. The returned score is that reranker's yes-probability. BM25 and cosine are the first stage. This is local, not on main.

We still cannot ask a question and get a Mamba-3 answer grounded in CDS code. The embed script has not been run on a corpus. Prompting, eval, and the app are still stubs.

Weighted total from the step table is 66.3%. Retrieved chunks are not yet fed to Mamba-3.

## Steps

Weights are the share of a functional assistant (question in, cited Mamba-3 answer out), not lines of code. Weighted total of the Done column: **66.3%**.

| Step | What | Status | Done | Weight |
|---|---|---|---|---|
| Model | Mamba-3 load and generate | Partial | 80% | 20 |
| 2 | Postgres + pgvector schema | Partial, local only | 70% | 5 |
| 3–4 | Ingest: crawl, filter, scan, chunk | Done | 90% | 18 |
| 5 | Embed and store | Partial, local only | 80% | 17 |
| 6 | `retrieve()`: BM25, then hybrid, then rerank | Partial, local only | 100% | 15 |
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

`index/schema.sql` is on main and safe to re-run. Still untracked: `docker-compose.yml`, `index/load_chunks.py`, `index/test_db.py`.

- Schema matches `contracts/chunk.schema.json`: chunk metadata, `embedding vector(1024)` for `Qwen/Qwen3-Embedding-0.6B`, `embedding_model` (both null or both set), `simple` tsvector, trigram index. No ANN index at init. Retrieval does not call the tsvector trigger.
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

### Step 5 — Embed and store (80%, local only)

The loader and the embed job are both written. Not on main. No corpus has been embedded yet.

- `index/load_chunks.py` upserts chunk text and metadata. It does not write `embedding`. It is on this branch, not on main.
- `index/embed.py` fills `embedding` and `embedding_model` for rows where `embedding` is null. Model is `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, L2-normalized. `--force` re-embeds every row. The schema check requires the vector and the model name together.
- Exact cosine until the table is large. No HNSW index.

### Step 6 — `retrieve()` (100%, local only)

v0 BM25, v1 hybrid, and v2 rerank are implemented locally. 100% is all three locked versions. Not on main.

- `retrieve(question, k=4, repos=None)` in `retrieval/` returns `contracts/retrieve.schema.json`. No HTTP route. Tests inject chunks, a fake connection, a fake embedder, or a fake reranker, so they do not need Postgres or a model download.
- v0: in-process BM25 (`k1=1.2`, `b=0.75`) over chunk text, with the code tokenizer (lowercase, split on non-letters, split camelCase / PascalCase / snake_case, keep the raw identifier). This is the keyword list. Postgres `ts_rank` and the `simple` tsvector are not used.
- v1: hybrid. Keyword and vector lists, up to 50 each, merged by reciprocal rank fusion (`1 / (60 + rank)`, rank starting at 1). Embedder is `Qwen/Qwen3-Embedding-0.6B` (1024-d). Exact cosine in process. A null embedding stays on the keyword list only. With no embeddings, the first stage follows BM25 and `timing_ms.embed` is 0. Otherwise `timing_ms.embed` is the query embed and `timing_ms.search` is the two lists plus the merge.
- v2: rerank. Rescore the top 20 first-stage hits with `Qwen/Qwen3-Reranker-0.6B`. `score` is the probability of yes. `timing_ms.rerank` is this stage. The response is at most `k` of those 20. Equal scores keep the first-stage order.
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

`ADVISOR_PRESENTATION.md` (October) marks Postgres, the loader, and chunking as complete infrastructure. Chunking and `index/schema.sql` are on main. The loader is still local. The embed job, hybrid first stage, and rerank are local and have not been run on a corpus. Use this file, not that deck, for current status.

## Shortest path from here

1. Load a JSONL into Postgres, then run `uv run python index/embed.py` so `chunks.embedding` is filled.
2. Ask `retrieve()` a question against those rows.
3. Build the numbered-source prompt from those chunks.
4. Generate with `mamba3.load_model` (SISO checkpoint, Llama-3.1 tokenizer).
5. Run one real corpus: pipeline or chunker, load the JSONL, embed, ask a question.

After that loop works: v2 rerank of the top 20, the `/generate` service, citation checking, the 50-question set, and the `/ask` app. Domain-adaptive training, the CUDA kernels, and a Transformer baseline are the research comparison, not blockers for the first answer.

## When you update this

1. Change the step's status, percent, and the "what exists" bullets.
2. Recompute the weighted total: sum of (done × weight) / 100.
3. Prepend a changelog line: date, what shipped, which step moved, and whether it is on main or still local.
4. Replace the "Team update" section with a few sentences of that delta. Leave the baseline table in git history; this file's diff is the comparison.
