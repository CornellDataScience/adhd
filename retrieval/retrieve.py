"""Load chunk rows and build a retrieve-contract response with BM25 (v0)."""

import os
import time
from collections.abc import Mapping, Sequence

from retrieval.bm25 import bm25_scores
from retrieval.tokenize import tokenize

DEFAULT_DATABASE_URL = "postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag"
MAX_K = 30

# Contract fields only. embedding and text_search stay out of ranking.
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
) -> dict:
    """Rank chunks for ``question`` with in-process BM25.

    Returns a dict matching ``contracts/retrieve.schema.json``. ``score`` is
    the BM25 score. ``timing_ms.embed`` and ``timing_ms.rerank`` are 0.
    ``timing_ms.search`` is this pass, in milliseconds.

    ``k`` defaults to 4, must be at least 1, and is clamped to 30. ``repos``
    limits the search to those repositories when it is set.

    Pass ``chunks`` or ``connection`` to avoid opening Postgres. Otherwise
    the URL comes from ``database_url``, ``DATABASE_URL``, or
    ``DEFAULT_DATABASE_URL``. Injected ``chunks`` win when both are passed.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > MAX_K:
        k = MAX_K

    started = time.perf_counter()
    loaded = _load_chunks(chunks, connection, database_url, repos)
    results = _rank(question, loaded, k)
    search_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "question": question,
        "results": results,
        "timing_ms": {"embed": 0, "search": search_ms, "rerank": 0},
    }


def _load_chunks(chunks, connection, database_url, repos) -> list[dict]:
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
    projected = [_public_chunk(row) for row in rows]
    if repos is None:
        return projected
    allowed = set(repos)
    return [chunk for chunk in projected if chunk["repo"] in allowed]


def _connect(database_url: str):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(database_url, row_factory=dict_row)


def _fetch_chunks(connection, repos: list[str] | None):
    if repos is not None and len(repos) == 0:
        return []
    sql = f"SELECT {', '.join(CHUNK_FIELDS)} FROM chunks"
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


def _rank(question: str, chunks: list[dict], k: int) -> list[dict]:
    if not chunks:
        return []
    query = tokenize(question)
    documents = [tokenize(chunk["text"]) for chunk in chunks]
    scores = bm25_scores(documents, query)
    # Highest score first. Equal scores break ties by chunk id.
    order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index]["id"]))
    results = []
    for index in order:
        score = scores[index]
        if score <= 0:
            break
        results.append({"rank": len(results) + 1, "score": float(score), "chunk": chunks[index]})
        if len(results) == k:
            break
    return results
