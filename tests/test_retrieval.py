import sys

import pytest

from retrieval.bm25 import B, K1, bm25_scores
from retrieval.retrieve import DEFAULT_DATABASE_URL, resolve_database_url, retrieve
from retrieval.tokenize import tokenize
from tests.test_contracts import validator

RESPONSE = validator("retrieve.schema.json", "response")


def test_load_model_is_kept_and_split():
    assert tokenize("load_model") == ["load_model", "load", "model"]


def test_camel_case_pascal_case_and_plain_words():
    assert tokenize("loadModel") == ["loadmodel", "load", "model"]
    assert tokenize("HTTPResponse") == ["httpresponse", "http", "response"]
    assert tokenize("backend") == ["backend"]
    assert tokenize("Running") == ["running"]


def test_digits_split_identifiers_and_are_not_tokens():
    assert tokenize("load2model") == ["load", "model"]
    assert "load_model" in tokenize("def load_model(model_id):")


def test_bm25_uses_standard_defaults():
    assert K1 == 1.2
    assert B == 0.75


def test_bm25_ranks_identifier_above_unrelated_chunk():
    matching = ["def", "load_model", "load", "model", "backend"]
    unrelated = ["forecast", "calls", "for", "rain"]
    unrelated_score, matching_score = bm25_scores([unrelated, matching], ["load_model", "load", "model"])
    assert matching_score > unrelated_score
    assert unrelated_score == 0


def test_bm25_prefers_exact_identifier_over_split_words():
    split_only = ["load", "the", "model", "from", "disk"]
    exact = ["def", "load_model", "load", "model"]
    split_score, exact_score = bm25_scores([split_only, exact], ["load_model"])
    assert exact_score > split_score
    assert split_score == 0


def make_chunk(number: int, repo: str, text: str) -> dict:
    return {
        "id": f"chunk-{number}",
        "repo": repo,
        "path": "mamba3/__init__.py",
        "start_line": 1,
        "end_line": 4,
        "symbol": "load_model",
        "language": "python",
        "kind": "code",
        "commit_sha": f"{number:040x}",
        "source_url": f"https://example.com/blob/{number}",
        "file_hash": f"{number:064x}",
        "token_count": 8,
        "text": text,
    }


def sample_chunks():
    return [
        make_chunk(1, "CornellDataScience/adhd", "The forecast calls for rain tomorrow."),
        make_chunk(2, "CornellDataScience/adhd", 'def load_model(model_id, backend="auto"):'),
        make_chunk(3, "CornellDataScience/FTM-Snake", "fn load_model() {}"),
        make_chunk(4, "CornellDataScience/adhd", "load the model from disk"),
    ]


class FakeCursor:
    def __init__(self, rows, log):
        self._rows = rows
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((sql, params))

    def fetchall(self):
        return list(self._rows)


class FakeConnection:
    def __init__(self, rows, log):
        self._rows = rows
        self._log = log
        self.closed = False

    def cursor(self):
        return FakeCursor(self._rows, self._log)

    def close(self):
        self.closed = True


class BoomConnection:
    def cursor(self):
        raise AssertionError("injected chunks must not open a cursor")

    def close(self):
        raise AssertionError("injected chunks must not close a connection")


def test_retrieve_response_is_schema_valid_and_ranked():
    response = retrieve(
        "How does load_model pick a backend?",
        k=4,
        repos=["CornellDataScience/adhd"],
        chunks=sample_chunks(),
        connection=BoomConnection(),
    )
    RESPONSE.validate(response)
    assert response["question"] == "How does load_model pick a backend?"
    assert response["timing_ms"]["embed"] == 0
    assert response["timing_ms"]["rerank"] == 0
    assert response["timing_ms"]["search"] >= 0
    assert [hit["chunk"]["repo"] for hit in response["results"]] == ["CornellDataScience/adhd"] * len(
        response["results"]
    )
    assert [hit["rank"] for hit in response["results"]] == list(range(1, len(response["results"]) + 1))
    assert response["results"][0]["chunk"]["id"] == "chunk-2"
    scores = [hit["score"] for hit in response["results"]]
    assert scores == sorted(scores, reverse=True)
    ids = {hit["chunk"]["id"] for hit in response["results"]}
    assert "chunk-3" not in ids
    assert "chunk-1" not in ids
    assert "chunk-4" in ids


def test_k_defaults_to_4_and_clamps_at_30():
    chunks = [
        make_chunk(i, "CornellDataScience/adhd", "load_model keeps this chunk relevant")
        for i in range(1, 36)
    ]
    defaulted = retrieve("load_model", chunks=chunks)
    assert len(defaulted["results"]) == 4
    assert [hit["rank"] for hit in defaulted["results"]] == [1, 2, 3, 4]
    clamped = retrieve("load_model", k=100, chunks=chunks)
    assert len(clamped["results"]) == 30
    assert [hit["rank"] for hit in clamped["results"]] == list(range(1, 31))
    RESPONSE.validate(clamped)


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError):
        retrieve("load_model", k=0, chunks=[])


def test_retrieve_from_fake_connection_filters_repos_without_ts_rank():
    log = []
    rows = sample_chunks()
    connection = FakeConnection(rows, log)
    response = retrieve(
        "load_model",
        k=4,
        repos=["CornellDataScience/FTM-Snake"],
        connection=connection,
    )
    RESPONSE.validate(response)
    assert connection.closed is False
    assert len(log) == 1
    sql, params = log[0]
    folded = sql.lower()
    assert "ts_rank" not in folded
    assert "tsvector" not in folded
    assert "text_search" not in folded
    assert "to_tsquery" not in folded
    assert params == (["CornellDataScience/FTM-Snake"],)
    assert [hit["rank"] for hit in response["results"]] == [1]
    assert response["results"][0]["chunk"]["id"] == "chunk-3"
    assert response["results"][0]["chunk"]["repo"] == "CornellDataScience/FTM-Snake"
    assert response["timing_ms"]["embed"] == 0
    assert response["timing_ms"]["rerank"] == 0
    assert response["timing_ms"]["search"] >= 0


def test_search_timing_is_measured(monkeypatch):
    ticks = iter([1.0, 1.5])
    retrieve_mod = sys.modules["retrieval.retrieve"]
    monkeypatch.setattr(retrieve_mod.time, "perf_counter", lambda: next(ticks))
    response = retrieve("load_model", chunks=[make_chunk(2, "CornellDataScience/adhd", "load_model")])
    assert response["timing_ms"] == {"embed": 0, "search": 500.0, "rerank": 0}


def test_database_url_override_does_not_connect(monkeypatch):
    assert DEFAULT_DATABASE_URL == "postgresql://adhd:dev_password_change_in_prod@localhost:5432/adhd_rag"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert resolve_database_url() == DEFAULT_DATABASE_URL
    monkeypatch.setenv("DATABASE_URL", "postgresql://unit-test/db")
    assert resolve_database_url() == "postgresql://unit-test/db"
    assert resolve_database_url("postgresql://arg/db") == "postgresql://arg/db"
