"""Citation supervised fine-tuning. Loss only on answer tokens. Full weights, no LoRA.

Each JSONL line is either:
  {"prompt": "...", "answer": "..."}
or
  {"question": "...", "retrieve_response": {...}, "answer": "..."}

    uv run python -m mamba3.train.sft \\
      --model /workspace/ckpts/cpt/last --data data/sft.jsonl --out /workspace/ckpts/sft --lr 2e-6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba3 import TOKENIZER_ID
from mamba3.train.checkpoint import load_trainable
from mamba3.train.loop import StepBatch, TrainConfig, train_steps
from prompting_eval.prompt import build_prompt


def _device(name: str | None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "prompt" in row and "answer" in row:
            rows.append({"prompt": row["prompt"], "answer": row["answer"]})
            continue
        if "retrieve_response" in row and "answer" in row:
            question = row.get("question") or row["retrieve_response"].get("question", "")
            prompt = build_prompt(row["retrieve_response"], question=question)
            rows.append({"prompt": prompt, "answer": row["answer"]})
            continue
        raise ValueError(f"need prompt/answer or retrieve_response/answer, got keys {sorted(row)}")
    return rows


def encode_example(tokenizer, prompt: str, answer: str, max_length: int) -> StepBatch:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = answer_ids + [eos]
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        # Keep the end of the prompt and the full answer when possible.
        overflow = len(ids) - max_length
        ids = ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)
    mask = [False] * prompt_len + [True] * (len(ids) - prompt_len)
    if len(ids) < 2:
        raise ValueError("example too short after tokenization")
    return StepBatch(
        input_ids=torch.tensor([ids], dtype=torch.long),
        loss_mask=torch.tensor([mask], dtype=torch.bool),
    )


def example_batches(examples: list[dict], tokenizer, max_length: int, epochs: int):
    for _ in range(epochs):
        for row in examples:
            yield encode_example(tokenizer, row["prompt"], row["answer"], max_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hub id or local CPT checkpoint directory")
    parser.add_argument("--data", type=Path, required=True, help="JSONL of citation examples")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "official", "torch"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--steps", type=int, default=None, help="Override epoch-derived step count")
    args = parser.parse_args()

    examples = load_examples(args.data)
    if not examples:
        raise SystemExit("no SFT examples")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    steps = args.steps or max(1, len(examples) * args.epochs)
    device = _device(args.device)
    backend = "torch" if device != "cuda" and args.backend == "auto" else args.backend
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    model = load_trainable(args.model, backend=backend, device=device, dtype=dtype)
    amp = "bf16" if device == "cuda" and dtype == torch.bfloat16 else ("fp16" if device == "cuda" else None)
    cfg = TrainConfig(
        lr=args.lr,
        min_lr=args.lr,
        total_steps=steps,
        warmup_steps=max(1, steps // 20),
        accumulate=1,
        amp_dtype=amp,
        save_every=0,
        log_every=1,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    records = train_steps(
        model,
        example_batches(examples, tokenizer, args.max_length, args.epochs),
        steps=steps,
        out_dir=args.out,
        log_path=args.out / "sft.jsonl",
        cfg=cfg,
        on_step=lambda row: print(json.dumps({"step": row["step"], "loss": row["loss"]}), flush=True),
    )
    print(json.dumps({"done_steps": len(records), "out": str(args.out / "last")}))


if __name__ == "__main__":
    main()
