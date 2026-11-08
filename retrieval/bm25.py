"""In-process BM25.

Lucene / Elasticsearch IDF, which stays non-negative:

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``k1=1.2`` and ``b=0.75`` are the standard defaults. Document length is the
number of tokenizer tokens. This module does not talk to Postgres and does
not call ``ts_rank``.
"""

import math
from collections import Counter
from collections.abc import Sequence

K1 = 1.2
B = 0.75


def bm25_scores(
    documents: Sequence[Sequence[str]],
    query: Sequence[str],
    *,
    k1: float = K1,
    b: float = B,
) -> list[float]:
    """Score each document against ``query``.

    ``documents`` and ``query`` are already tokenized with the same tokenizer.
    A repeated query term contributes once per occurrence. An empty query, an
    empty corpus, or a term that appears in no document contributes nothing.
    """
    n_docs = len(documents)
    if n_docs == 0:
        return []

    term_freqs = [Counter(document) for document in documents]
    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / n_docs
    scores = [0.0] * n_docs
    if not query or average_length == 0:
        return scores

    document_freq: Counter[str] = Counter()
    for term_freq in term_freqs:
        document_freq.update(term_freq.keys())

    for term, query_tf in Counter(query).items():
        df = document_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for index, term_freq in enumerate(term_freqs):
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            length_norm = 1.0 - b + b * (lengths[index] / average_length)
            numerator = freq * (k1 + 1.0)
            scores[index] += query_tf * idf * numerator / (freq + k1 * length_norm)
    return scores
