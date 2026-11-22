import math

import pytest

from retrieval.embed import DIMENSIONS, MODEL_ID, TASK, embed_documents, embed_query, format_query


def test_query_uses_the_instruction_wrapper_and_documents_do_not():
    assert MODEL_ID == "Qwen/Qwen3-Embedding-0.6B"
    assert DIMENSIONS == 1024
    wrapped = format_query("How does load_model pick a backend?")
    assert wrapped == f"Instruct: {TASK}\nQuery:How does load_model pick a backend?"
    assert not wrapped.startswith("How does")


@pytest.mark.slow
def test_real_embedder_is_unit_norm_and_query_differs_from_document():
    text = "def load_model(model_id): pass"
    query = embed_query(text)
    document = embed_documents([text])[0]
    assert len(query) == DIMENSIONS
    assert len(document) == DIMENSIONS
    assert math.isclose(math.sqrt(sum(value * value for value in query)), 1.0, abs_tol=1e-3)
    assert math.isclose(math.sqrt(sum(value * value for value in document)), 1.0, abs_tol=1e-3)
    assert query != document
