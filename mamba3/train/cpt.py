"""Continued pretraining over a packed corpus. Full weights, no LoRA.

    uv run python -m mamba3.train.cpt \\
      --model siso-187m --data /workspace/packed --out /workspace/ckpts/cpt \\
      --token-budget 500000000 --batch-size 1 --accumulate 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from mamba3.train.checkpoint import load_trainable
from mamba3.train.data import PackedCorpus
from mamba3.train.loop import StepBatch, TrainConfig, train_steps


def _device(name: str | None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(name: str | None, device: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def packed_batches(corpus: PackedCorpus, batch_size: int, seed: int = 0):
    order = list(range(len(corpus)))
    rng = random.Random(seed)
    while True:
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            if len(chunk) < batch_size:
                continue
            yield StepBatch(input_ids=corpus.batch(chunk))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="siso-187m")
    parser.add_argument("--data", type=Path, required=True, help="Directory from mamba3.train.pack")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "official", "torch"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--token-budget", type=int, required=True, help="Stop after this many training tokens")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = PackedCorpus(args.data)
    tokens_per_step = args.batch_size * args.accumulate * (corpus.seq_len - 1)
    steps = max(1, args.token_budget // max(1, tokens_per_step))
    warmup = max(1, steps // 100)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    load_dtype = torch.float32 if device == "cuda" and dtype == torch.float16 else dtype
    backend = "torch" if device != "cuda" and args.backend == "auto" else args.backend
    model = load_trainable(args.model, backend=backend, device=device, dtype=load_dtype)

    amp = "bf16" if device == "cuda" and dtype == torch.bfloat16 else ("fp16" if device == "cuda" else None)
    cfg = TrainConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        total_steps=steps,
        warmup_steps=warmup,
        accumulate=args.accumulate,
        amp_dtype=amp,
        save_every=args.save_every,
        log_every=1,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"n_blocks": len(corpus), "seq_len": corpus.seq_len, "steps": steps, "tokens_per_step": tokens_per_step}))
    records = train_steps(
        model,
        packed_batches(corpus, args.batch_size, seed=args.seed),
        steps=steps,
        out_dir=args.out,
        log_path=args.out / "train.jsonl",
        cfg=cfg,
        on_step=lambda row: print(json.dumps(row), flush=True) if row["step"] % 10 == 0 else None,
    )
    print(json.dumps({"done_steps": len(records), "out": str(args.out / "last")}))


if __name__ == "__main__":
    main()
