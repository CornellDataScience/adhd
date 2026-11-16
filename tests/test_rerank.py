import pytest

from retrieval.embed import TASK
from retrieval.rerank import MODEL_ID, PREFIX, SUFFIX, format_document, rerank


def test_rerank_prompt_uses_the_shared_code_task():
    assert MODEL_ID == "Qwen/Qwen3-Reranker-0.6B"
    text = format_document("load_model", "def load_model(): pass")
    assert text == f"<Instruct>: {TASK}\n<Query>: load_model\n<Document>: def load_model(): pass"
    assert PREFIX.startswith("<|im_start|>system\n")
    assert 'only be "yes" or "no"' in PREFIX
    assert SUFFIX.endswith("<think>\n\n</think>\n\n")


@pytest.mark.slow
def test_real_reranker_prefers_the_matching_chunk():
    scores = rerank(
        "How does load_model pick a backend?",
        [
            "The forecast calls for rain tomorrow.",
            'def load_model(model_id, backend="auto"):',
        ],
    )
    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[1] > scores[0]
