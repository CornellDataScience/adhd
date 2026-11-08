
import os
import time
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from mamba3 import TOKENIZER_ID, load_model


# Start with the smallest Mamba-3 checkpoint
MODEL_NAME = os.getenv("MODEL_NAME", "siso-187m")

# Model and tokenizer are loaded once at startup
model = None
tokenizer = None

# Prevent simultaneous generation on the same model
generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=256)


class GenerateResponse(BaseModel):
    answer: str
    model: str
    generation_latency_ms: float
    output_tokens: int
    tokens_per_second: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = load_model(MODEL_NAME, backend="torch")
    model.eval()

    print("Mamba-3 model is ready!")

    yield

    model = None
    tokenizer = None


app = FastAPI(
    title="CDS Mamba-3 Inference Service",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model": MODEL_NAME
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    with generation_lock:

        device = next(model.parameters()).device

        # Convert the prompt into tokens
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.to(device)

        # Check model context length
        if input_ids.shape[1] + request.max_new_tokens > 2048:
            raise HTTPException(
                status_code=400,
                detail="Prompt exceeds the model context limit"
            )

        start_time = time.perf_counter()

        # Generate text using Mamba-3
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_length=(
                    input_ids.shape[1]
                    + request.max_new_tokens
                ),
                top_k=1
            )

        # Synchronize GPU before recording elapsed time
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time

        # Only decode the newly generated tokens
        new_tokens = output[0, input_ids.shape[1]:]

        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        token_count = len(new_tokens)

        return GenerateResponse(
            answer=answer,
            model=MODEL_NAME,
            generation_latency_ms=round(
                elapsed * 1000, 2
            ),
            output_tokens=token_count,
            tokens_per_second=round(
                token_count / elapsed, 2
            ) if elapsed > 0 else 0.0
        )
