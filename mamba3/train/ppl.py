"""Held-out perplexity on packed blocks or a text file.

    uv run python -m mamba3.train.ppl --model siso-187m --data /workspace/packed-holdout
    uv run python -m mamba3.train.ppl --model /workspace/ckpts/cpt/last --text-file holdout.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba3 import TOKENIZER_ID, load_model
from mamba3.train.data import PackedCorpus
from mamba3.train.logits import model_logits
from mamba3.train.loss import next_token_loss


def _device(name: str | None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def perplexity_on_corpus(model, corpus: PackedCorpus, *, max_blocks: int | None = None) -> dict:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    n = len(corpus) if max_blocks is None else min(len(corpus), max_blocks)
    for index in range(n):
        ids = corpus[index].unsqueeze(0).to(device)
        logits = model_logits(model, ids)
        loss, count = next_token_loss(logits, ids)
        total_loss += float(loss.item()) * count
        total_tokens += count
    mean = total_loss / max(1, total_tokens)
    return {"n_blocks": n, "tokens": total_tokens, "loss": mean, "perplexity": math.exp(mean)}


@torch.no_grad()
def perplexity_on_text(model, tokenizer, text: str, seq_len: int = 2048) -> dict:
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < 2:
        raise ValueError("text too short")
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start : start + seq_len]
        if len(chunk) < 2:
            break
        tensor = torch.tensor([chunk], dtype=torch.long, device=device)
        logits = model_logits(model, tensor)
        loss, count = next_token_loss(logits, tensor)
        total_loss += float(loss.item()) * count
        total_tokens += count
    mean = total_loss / max(1, total_tokens)
    return {"tokens": total_tokens, "loss": mean, "perplexity": math.exp(mean)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, help="Packed corpus directory")
    parser.add_argument("--text-file", type=Path, help="Single UTF-8 text file")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--backend", choices=["auto", "official", "torch"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    args = parser.parse_args()
    if bool(args.data) == bool(args.text_file):
        raise SystemExit("Pass exactly one of --data or --text-file")

    device = _device(args.device)
    backend = "torch" if device != "cuda" and args.backend == "auto" else args.backend
    dtype = torch.float32 if device != "cuda" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = load_model(args.model, backend=backend, device=device, dtype=dtype)
    model.eval()

    if args.data:
        result = perplexity_on_corpus(model, PackedCorpus(args.data), max_blocks=args.max_blocks)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        result = perplexity_on_text(model, tokenizer, args.text_file.read_text(encoding="utf-8"))
    result["model"] = args.model
    result["device"] = device
    print(json.dumps(result))


if __name__ == "__main__":
    main()
