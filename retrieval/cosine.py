"""Exact cosine over stored vectors.

Vectors from ``retrieval.embed`` are L2-normalized, so cosine is a dot product.
Missing vectors are left out of the list. This does not call Postgres.
"""

from collections.abc import Sequence


def top_cosine(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float] | None],
    query: Sequence[float],
    limit: int,
) -> list[str]:
    """Up to ``limit`` ids, highest cosine first. Ties break by id.

    A vector that is missing or a different length than ``query`` is skipped.
    """
    if limit < 1 or not query:
        return []
    width = len(query)
    scored: list[tuple[float, str]] = []
    for chunk_id, vector in zip(ids, vectors, strict=True):
        if vector is None or len(vector) != width:
            continue
        score = sum(left * right for left, right in zip(vector, query, strict=True))
        scored.append((score, chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in scored[:limit]]
