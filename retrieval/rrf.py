"""Reciprocal rank fusion.

``score = sum 1 / (60 + rank)`` with rank starting at 1. The constant is the
one fixed in Cormack, Clarke, and Büttcher, SIGIR 2009, and Elasticsearch's
default ``rank_constant``. Fusion uses rank, not the raw BM25 or cosine score.
"""

from collections.abc import Sequence

RANK_CONSTANT = 60


def fuse(
    ranked_lists: Sequence[Sequence[str]],
    *,
    rank_constant: int = RANK_CONSTANT,
) -> list[tuple[str, float]]:
    """Merge ranked id lists. Higher score first. Equal scores break ties by id.

    A chunk in only one list still gets that list's term. A repeated id in a
    single list counts once, at its first rank.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        rank = 0
        for item_id in ranked:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
