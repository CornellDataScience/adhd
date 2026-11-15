
from inference.rag_prompt import (
    build_rag_prompt,
    extract_sources,
)


def test_retrieval_to_prompt():
    results = [
        {
            "rank": 1,
            "score": 0.91,
            "chunk": {
                "id": "test-chunk-1",
                "repo": "CornellDataScience/adhd",
                "path": "mamba3/__init__.py",
                "start_line": 30,
                "end_line": 47,
                "source_url": "https://github.com/example",
                "text": "load_model chooses a model backend."
            },
        }
    ]

    prompt = build_rag_prompt(
        "How does load_model work?",
        results,
    )
    sources = extract_sources(results)

    assert "[1]" in prompt
    assert "load_model chooses a model backend" in prompt
    assert sources[0]["rank"] == 1
    assert sources[0]["path"] == "mamba3/__init__.py"
    assert sources[0]["chunk_id"] == "test-chunk-1"
