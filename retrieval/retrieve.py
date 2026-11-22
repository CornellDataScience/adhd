"""Load chunk rows and build a retrieve-contract response.

The first stage is BM25, or BM25 fused with exact cosine when any chunk has
an embedding. The top 20 of that list are rescored by the Qwen reranker.
"""

import os
import time
from collections.abc import Mapping, Sequence

from retrieval.bm25 import bm25_scores
from retrieval.cosine import top_cosine
from retrieval.rrf import fuse
from retrieval.tokenize import tokenize

DEFAULT_DATABASE_URL = "postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag"
MAX_K = 30
KEYWORD_LIMIT = 50
VECTOR_LIMIT = 50
RERANK_LIMIT = 20

# Public response fields. The embedding column is read for cosine and then dropped.
CHUNK_FIELDS = (
    "id",
    "repo",
    "path",
    "start_line",
    "end_line",
    "symbol",
    "language",
    "kind",
    "commit_sha",
    "source_url",
    "file_hash",
    "token_count",
    "text",
)


def resolve_database_url(database_url: str | None = None) -> str:
    """Argument, then ``DATABASE_URL``, then the local adhd_rag default."""
    if database_url:
        return database_url
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def retrieve(
    question: str,
    k: int = 4,
    repos: list[str] | None = None,
    *,
    chunks: list[dict] | None = None,
    connection=None,
    database_url: str | None = None,
    embedder=None,
    reranker=None,
) -> dict:
    """Rank chunks for ``question``.

    Returns a dict matching ``contracts/retrieve.schema.json``. ``k`` defaults
    to 4, must be at least 1, and is clamped to 30. ``repos`` limits the search
    to those repositories when it is set.

    The first stage keeps up to 50 keyword hits and, when any chunk has an
    embedding, up to 50 cosine hits, merged by reciprocal rank fusion. With
    no embeddings the question is not embedded and that stage follows BM25
    order. The top 20 fused hits are rescored, ``score`` is that reranker
    score, and at most ``k`` of them are returned. ``timing_ms.embed`` is the
    query embedding, ``timing_ms.search`` is the two lists plus the merge, and
    ``timing_ms.rerank`` is the rescore. An empty candidate list skips the
    reranker and leaves ``timing_ms.rerank`` at 0.

    Pass ``chunks`` or ``connection`` to avoid opening Postgres. Otherwise the
    URL comes from ``database_url``, ``DATABASE_URL``, or
    ``DEFAULT_DATABASE_URL``. Injected ``chunks`` win when both are passed.
    ``embedder`` maps the question to a vector. When it is omitted and a
    vector list is needed, ``retrieval.embed.embed_query`` loads
    ``Qwen/Qwen3-Embedding-0.6B``. ``reranker`` maps the question and the
    candidate texts to one score each. When it is omitted,
    ``retrieval.rerank.rerank`` loads ``Qwen/Qwen3-Reranker-0.6B``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > MAX_K:
        k = MAX_K

    started = time.perf_counter()
    loaded = _load_chunks(chunks, connection, database_url, repos)
    rows = [chunk for chunk, _vector in loaded]
    vectors = [vector for _chunk, vector in loaded]
    embed_ms = 0
    query_vector = None
    if any(vector is not None for vector in vectors):
        embed_started = time.perf_counter()
        query_vector = _embed_question(question, embedder)
        embed_ms = round((time.perf_counter() - embed_started) * 1000, 3)
        search_started = time.perf_counter()
    else:
        search_started = started

    keyword_ids = _keyword_ids(question, rows)
    if query_vector is None:
        vector_ids: list[str] = []
    else:
        vector_ids = top_cosine([chunk["id"] for chunk in rows], vectors, query_vector, VECTOR_LIMIT)
    by_id = {chunk["id"]: chunk for chunk in rows}
    pool = []
    for chunk_id, _score in fuse([keyword_ids, vector_ids]):
        pool.append(by_id[chunk_id])
        if len(pool) == RERANK_LIMIT:
            break
    search_ms = round((time.perf_counter() - search_started) * 1000, 3)

    rerank_ms = 0
    if pool:
        rerank_started = time.perf_counter()
        scores = _rerank_pool(question, pool, reranker)
        rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 3)
        # Equal reranker scores keep the first-stage order.
        order = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
        ranked = [pool[index] for index in order]
        ranked_scores = [scores[index] for index in order]
    else:
        ranked = []
        ranked_scores = []
    results = [
        {"rank": index + 1, "score": float(ranked_scores[index]), "chunk": chunk}
        for index, chunk in enumerate(ranked[:k])
    ]
    return {
        "question": question,
        "results": results,
        "timing_ms": {"embed": embed_ms, "search": search_ms, "rerank": rerank_ms},
    }


def _rerank_pool(question: str, pool: list[dict], reranker) -> list[float]:
    documents = [chunk["text"] for chunk in pool]
    if reranker is None:
        from retrieval.rerank import rerank

        scores = rerank(question, documents)
    else:
        scores = [float(score) for score in reranker(question, documents)]
    if len(scores) != len(documents):
        raise ValueError("reranker must return one score per document")
    return scores


def _embed_question(question: str, embedder) -> list[float]:
    if embedder is None:
        from retrieval.embed import embed_query

        return embed_query(question)
    return [float(item) for item in embedder(question)]


def _load_chunks(chunks, connection, database_url, repos) -> list[tuple[dict, list[float] | None]]:
    if chunks is not None:
        rows = chunks
    else:
        owns_connection = connection is None
        if owns_connection:
            connection = _connect(resolve_database_url(database_url))
        try:
            rows = _fetch_chunks(connection, repos)
        finally:
            if owns_connection:
                connection.close()
    prepared = [(_public_chunk(row), _row_vector(row)) for row in rows]
    if repos is None:
        return prepared
    allowed = set(repos)
    return [item for item in prepared if item[0]["repo"] in allowed]


def _connect(database_url: str):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(database_url, row_factory=dict_row)


def _fetch_chunks(connection, repos: list[str] | None):
    if repos is not None and len(repos) == 0:
        return []
    sql = f"SELECT {', '.join(CHUNK_FIELDS)}, embedding FROM chunks"
    params = None
    if repos is not None:
        sql += " WHERE repo = ANY(%s)"
        params = (list(repos),)
    with connection.cursor() as cur:
        if params is None:
            cur.execute(sql)
        else:
            cur.execute(sql, params)
        return list(cur.fetchall())


def _row_vector(row: Mapping | Sequence) -> list[float] | None:
    if not isinstance(row, Mapping) or "embedding" not in row:
        return None
    return _as_vector(row.get("embedding"))


def _as_vector(value) -> list[float] | None:
    """Accept a list, a pgvector ``Vector``, or the text form Postgres returns."""
    if value is None:
        return None
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text.strip():
            return None
        return [float(part) for part in text.split(",")]
    try:
        vector = [float(item) for item in value]
    except TypeError:
        return None
    return vector or None


def _public_chunk(row: Mapping | Sequence) -> dict:
    if isinstance(row, Mapping):
        data = row
    else:
        data = dict(zip(CHUNK_FIELDS, row, strict=True))
    chunk = {field: data[field] for field in CHUNK_FIELDS}
    chunk["start_line"] = int(chunk["start_line"])
    chunk["end_line"] = int(chunk["end_line"])
    chunk["token_count"] = int(chunk["token_count"])
    return chunk


def _keyword_ids(question: str, chunks: list[dict]) -> list[str]:
    """Up to 50 chunk ids with a positive BM25 score. Highest score first.

    Equal scores break ties by chunk id. Rank in this list starts at 1 when
    it is fused.
    """
    if not chunks:
        return []
    query = tokenize(question)
    documents = [tokenize(chunk["text"]) for chunk in chunks]
    scores = bm25_scores(documents, query)
    order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index]["id"]))
    ids = []
    for index in order:
        if scores[index] <= 0:
            break
        ids.append(chunks[index]["id"])
        if len(ids) == KEYWORD_LIMIT:
            break
    return ids
