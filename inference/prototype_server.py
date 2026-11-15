
"""FastAPI inference server for the CDS Mamba-3 RAG assistant."""

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from mamba3 import TOKENIZER_ID, load_model
from inference.rag_prompt import build_rag_prompt, extract_sources


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "siso-187m")
MAX_CONTEXT_TOKENS = 2048

model = None
tokenizer = None

# Only one generation request executes at a time.
generation_lock = threading.Lock()


# --------------------------------------------------
# Retrieval contract models
# --------------------------------------------------

class RetrievalChunk(BaseModel):
    id: str
    repo: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: str | None
    language: str

    kind: Literal[
        "code",
        "doc",
        "readme",
        "file_tree",
        "file_summary",
    ]

    commit_sha: str
    source_url: str
    file_hash: str
    token_count: int = Field(ge=1)
    text: str

    model_config = {"extra": "forbid"}


class RetrievalResult(BaseModel):
    rank: int = Field(ge=1)
    score: float
    chunk: RetrievalChunk

    model_config = {"extra": "forbid"}


class RetrievalTiming(BaseModel):
    embed: float = Field(ge=0)
    search: float = Field(ge=0)
    rerank: float = Field(ge=0)

    model_config = {"extra": "forbid"}


# --------------------------------------------------
# API request and response models
# --------------------------------------------------

class GenerateRequest(BaseModel):
    question: str = Field(min_length=1)

    results: list[RetrievalResult] = Field(
        max_length=30
    )

    timing_ms: RetrievalTiming

    max_new_tokens: int = Field(
        default=64,
        ge=1,
        le=256,
    )

    model_config = {"extra": "forbid"}


class SourceReference(BaseModel):
    rank: int
    chunk_id: str
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    commit_sha: str
    source_url: str


class GenerateResponse(BaseModel):
    answer: str
    model: str

    sources: list[SourceReference]

    retrieval_timing_ms: dict[str, float]

    generation_latency_ms: float
    output_tokens: int
    tokens_per_second: float


# --------------------------------------------------
# Model startup and shutdown
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    print(f"Loading Mamba-3 model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID
    )

    # Portable backend works on CPU and Apple MPS.
    # CUDA deployment can be configured separately.
    model = load_model(
        MODEL_NAME,
        backend="torch",
    )

    model.eval()

    print("Mamba-3 model loaded successfully!")

    yield

    model = None
    tokenizer = None

    print("Inference server stopped.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="CDS Mamba-3 RAG Inference Service",
    description=(
        "Model-serving API for Mamba-3 with "
        "ranked CDS retrieval context."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": (
            "ready"
            if model is not None and tokenizer is not None
            else "loading"
        ),
        "model": MODEL_NAME,
        "backend": "torch",
    }


# --------------------------------------------------
# Device synchronization
# --------------------------------------------------

def synchronize_device(device):
    """Wait for outstanding GPU operations to finish."""

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# Generate endpoint
# --------------------------------------------------

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    # 1. Verify model availability
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Mamba-3 model is not ready.",
        )

    # 2. Validate the question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # 3. Validate retrieval results
    if not request.results:
        raise HTTPException(
            status_code=400,
            detail="No retrieved results were provided.",
        )

    # Retrieval ranks should be 1, 2, 3, ...
    ranks = [
        result.rank
        for result in request.results
    ]

    expected_ranks = list(
        range(1, len(ranks) + 1)
    )

    if ranks != expected_ranks:
        raise HTTPException(
            status_code=400,
            detail=(
                "Retrieval results must be ranked "
                "consecutively starting at 1."
            ),
        )

    # 4. Convert validated results to dictionaries
    results = [
        result.model_dump()
        for result in request.results
    ]

    # 5. Construct the RAG prompt
    prompt = build_rag_prompt(
        question=request.question,
        results=results,
    )

    # 6. Preserve retrieval source metadata
    sources = extract_sources(results)

    # 7. Perform model generation
    with generation_lock:

        device = next(
            model.parameters()
        ).device

        # Tokenize the RAG prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.to(device)

        input_token_count = input_ids.shape[1]

        total_length = (
            input_token_count
            + request.max_new_tokens
        )

        # 8. Enforce the context-window limit
        if total_length > MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The retrieved context exceeds "
                    "the 2048-token model context window. "
                    "Reduce the number of chunks or "
                    "the requested output length."
                ),
            )

        # 9. Synchronize before timing
        synchronize_device(device)

        start_time = time.perf_counter()

        # 10. Generate Mamba-3 response
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_length=total_length,
                top_k=1,
            )

        # 11. Synchronize after generation
        synchronize_device(device)

        elapsed = time.perf_counter() - start_time

        # 12. Extract generated tokens
        generated_tokens = output[
            0,
            input_token_count:
        ]

        # 13. Decode generated text
        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        output_token_count = len(
            generated_tokens
        )

        # 14. Calculate performance
        latency_ms = elapsed * 1000

        tokens_per_second = (
            output_token_count / elapsed
            if elapsed > 0
            else 0.0
        )

        # 15. Return response
        return GenerateResponse(
            answer=answer,
            model=MODEL_NAME,
            sources=sources,
            retrieval_timing_ms=(
                request.timing_ms.model_dump()
            ),
            generation_latency_ms=round(
                latency_ms, 2
            ),
            output_tokens=output_token_count,
            tokens_per_second=round(
                tokens_per_second, 2
            ),
        )
