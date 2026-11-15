
"""
End-to-end CDS RAG demonstration.

Flow:
Question -> BM25 retrieval -> FastAPI -> Mamba-3 -> Answer
"""

import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from retrieval.retrieve import retrieve


# Project paths and API configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "samples" / "chunks.jsonl"

INFERENCE_URL = "http://127.0.0.1:8000/generate"


def load_chunks():
    """Load repository chunks from the sample JSONL file."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Sample chunks not found: {CHUNKS_FILE}"
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks.")
    return chunks


def main():
    # 1. Choose a question about the CDS codebase
    question = "How does load_model pick a backend?"

    print("\n=== USER QUESTION ===")
    print(question)

    # 2. Load sample documents
    chunks = load_chunks()

    # 3. Run your teammate's actual BM25 retrieval
    retrieved = retrieve(
        question=question,
        k=2,
        chunks=chunks,
    )

    print("\n=== BM25 RETRIEVAL ===")

    results = retrieved["results"]

    if not results:
        print("No matching chunks found.")
        return

    for result in results:
        chunk = result["chunk"]

        print(
            f"[{result['rank']}] "
            f"{chunk['repo']}/{chunk['path']}"
        )
        print(f"Score: {result['score']:.4f}")
        print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
        print()

    # 4. Prepare the request for your FastAPI service
    payload = {
        "question": retrieved["question"],
        "results": retrieved["results"],
        "timing_ms": retrieved["timing_ms"],
        "max_new_tokens": 64,
    }

    request = Request(
        url=INFERENCE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # 5. Send retrieved context to Mamba-3
    print("\n=== GENERATING ANSWER ===")

    try:
        with urlopen(request, timeout=180) as response:
            generation = json.load(response)

    except HTTPError as error:
        print(f"API error: HTTP {error.code}")
        print(error.read().decode("utf-8"))
        return

    except URLError as error:
        print(f"Cannot reach inference server: {error}")
        print("Make sure FastAPI is running on port 8000.")
        return

    # 6. Display generated answer
    print("\n=== MAMBA-3 ANSWER ===")
    print(generation["answer"])

    # 7. Display retrieved sources
    print("\n=== RETRIEVED SOURCES ===")

    for source in generation["sources"]:
        print(
            f"[{source['rank']}] "
            f"{source['repo']}/{source['path']}"
        )
        print(source["source_url"])
        print()

    # 8. Display inference metrics
    print("\n=== PERFORMANCE ===")

    retrieval_ms = sum(
        retrieved["timing_ms"].values()
    )

    generation_ms = generation[
        "generation_latency_ms"
    ]

    print(f"Retrieval latency: {retrieval_ms:.2f} ms")
    print(f"Generation latency: {generation_ms:.2f} ms")
    print(
        f"Combined measured latency: "
        f"{retrieval_ms + generation_ms:.2f} ms"
    )
    print(
        f"Output tokens: {generation['output_tokens']}"
    )
    print(
        f"Tokens/sec: {generation['tokens_per_second']}"
    )

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
