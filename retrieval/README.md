# Retrieval

Decision record for step 6. Locked Nov 8, 2026. v0 is implemented. v1 and v2 are not. Build the remaining versions in order. Each version is the same function and the same response, so a later version replaces the ranker instead of adding an API.

`RETRIEVAL_ARCHITECTURE.md` and `RETRIEVAL_OPTIONS_FOR_ADVISOR.md` are background. This file is the decision.

## Interface

`retrieve(question, k=4, repos=None)` lives in `retrieval/`. It returns a response that matches `contracts/retrieve.schema.json`. There is no HTTP route.

`k` defaults to 4 and is at most 30. `repos` limits the search to those repositories when it is set. Every version honors that filter.

The response carries the question, `results` best-first, and `timing_ms`. Each result has `rank` (starting at 1; that number is the citation a later prompt will use), `score`, and `chunk` (a row matching `contracts/chunk.schema.json`). The contract describes `score` as a reranker score. Until v2, `score` is the score of the ranker that is actually running. `timing_ms` has `embed`, `search`, and `rerank`.

Chunks are the rows already loaded into local Postgres from `samples/chunks.jsonl`. Retrieval reads that text. It does not call the `english` `tsvector` trigger. The trigger stays in the database.

`vector(1536)` was never a model choice. That width is OpenAI `text-embedding-3-small`. The schema comment had paired it with Qwen3-Embedding-0.6B. Nothing has been embedded. The column is `vector(1024)` for the model in v1.

`psycopg` is a project dependency, and `retrieval` is one of the hatch wheel packages.

## v0 — BM25 and a code tokenizer (implemented)

This is the first version. Postgres `ts_rank` is out of the plan.

BM25 is the lexical baseline in Lucene, Elasticsearch, and TREC. Use the standard defaults `k1=1.2` and `b=0.75`.

Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*.

Postgres `ts_rank` does not use corpus-wide term statistics, so it is not that baseline. Do not put it back in as v0. The existing `english` `tsvector` trigger stays; retrieval does not use it.

PostgreSQL documentation, 12.3, Controlling Text Search ([textsearch-controls](https://www.postgresql.org/docs/current/textsearch-controls.html)).

Tokenizer, applied to the question and to each chunk:

- Lowercase.
- Split on non-letters.
- Split camelCase, PascalCase, and snake_case.
- Keep the raw identifier as its own token, so `load_model` still matches exactly.
- No English stemmer.

Run BM25 in process over the chunk text already stored in Postgres. At the sample size that is the simple version. `timing_ms.embed` and `timing_ms.rerank` are 0. `timing_ms.search` is this pass. `score` is the BM25 score.

## v1 — hybrid (not implemented)

Take a keyword list and a vector list, up to 50 hits each. The keyword list is the v0 BM25 ranker. The vector list is cosine over the stored embeddings. Merge the lists with reciprocal rank fusion and return `k`.

`score = sum over lists of 1 / (60 + rank)`, with rank starting at 1. A chunk that appears in only one list still gets that list's term.

`k=60` is the constant fixed in Cormack, Clarke, and Büttcher, SIGIR 2009, and it is Elasticsearch's default `rank_constant`. Fusion uses rank, not the raw score, because BM25 and cosine are different scales.

Embedder: `Qwen/Qwen3-Embedding-0.6B`. 1024 dimensions, cosine similarity, Apache-2.0. About 75.4 on MTEB-Code. It is laptop-sized; the 4B and 8B variants are not the choice.

Store 1024-d vectors. For the sample, scan with exact cosine. When the table is large, add an HNSW index. Do not create an IVFFlat index with `lists = 100`. That setting assumes on the order of 100,000 rows.

`timing_ms.embed` is the time to embed the question. `timing_ms.rerank` stays 0. `timing_ms.search` is the two lists plus the merge. `score` is the fusion score.

## v2 — rerank (not implemented)

Rescore the top 20 hybrid hits with `Qwen/Qwen3-Reranker-0.6B` and return `k`.

A cross-encoder second stage is the standard after a cheap first stage.

Nogueira and Cho, *Passage Re-ranking with BERT*, 2019.

This reranker is about 73.4 on MTEB-Code. `BGE-reranker-v2-m3` is about 41 on that set. `ms-marco-MiniLM-L-6-v2` is a web-passage model from the MS MARCO track, not a code reranker.

The Qwen reranker paper scores candidates from a top-100 pool. 20 is the laptop cap locked here, not that paper's protocol.

`score` becomes the reranker score. `timing_ms.rerank` is this stage.

## Not in these versions

A FastAPI `/retrieve` route, Mamba generation, citation checking, and the 50-question eval. Those consume this function's return value.

## When a version lands

Say so in that version's heading, and prepend a line to the changelog in `STATUS.md`.
