"""Prove one machine can train Mamba-3: forward, backward, save, reload.

    # On the GPU host after uv sync --extra cuda:
    uv run python -m mamba3.train.smoke --model siso-187m --steps 10 --out /workspace/smoke-ckpt

    # Tiny random model (no Hub download; for CI / laptop wiring checks):
    uv run python -m mamba3.train.smoke --tiny --steps 5 --seq-len 32 --out /tmp/smoke-tiny
"""

from __future__ import annotations

import argparse
import json
from itertools import repeat
from pathlib import Path

import torch

from mamba3 import load_model
from mamba3.train.checkpoint import load_trainable, save_checkpoint
from mamba3.train.logits import model_logits
from mamba3.train.loop import StepBatch, TrainConfig, train_steps


TINY_CONFIG = {
    "d_model": 64,
    "d_intermediate": 128,
    "n_layer": 2,
    "vocab_size": 128,
    "ssm_cfg": {"layer": "Mamba3", "d_state": 32, "headdim": 16, "chunk_size": 16},
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True,
}


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


def build_tiny(device: str, dtype: torch.dtype):
    from mamba3.mamba3_torch import Mamba3LMHeadModel

    model = Mamba3LMHeadModel(TINY_CONFIG, device=device, dtype=dtype)
    model.train()
    return model


def random_batches(batch_size: int, seq_len: int, vocab: int, device: str):
    while True:
        ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        yield StepBatch(input_ids=ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="siso-187m")
    parser.add_argument("--tiny", action="store_true", help="Use a random tiny model; no Hub download")
    parser.add_argument("--backend", choices=["auto", "official", "torch"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128, help="Use 2048 on the GPU host when memory allows")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.tiny:
        model = build_tiny(device, dtype)
        vocab = TINY_CONFIG["vocab_size"]
        backend_used = "tiny"
    else:
        # load_model returns eval(); load_trainable flips to train.
        if args.backend == "auto" and device == "cpu":
            # Avoid trying official kernels when there is no CUDA.
            backend = "torch"
        else:
            backend = args.backend
        model = load_trainable(args.model, backend=backend, device=device, dtype=dtype)
        vocab = int(getattr(model.config, "vocab_size", None) or model.config["vocab_size"])
        backend_used = backend

    # One forward shape check before the loop.
    probe = torch.randint(0, vocab, (args.batch_size, min(args.seq_len, 32)), device=device)
    logits = model_logits(model, probe)
    print(json.dumps({"logits_shape": list(logits.shape), "device": device, "backend": backend_used}))
    if list(logits.shape) != [args.batch_size, probe.shape[1], logits.shape[-1]]:
        raise SystemExit(f"unexpected logits shape {tuple(logits.shape)}")

    amp = "bf16" if device == "cuda" and dtype == torch.bfloat16 else ("fp16" if device == "cuda" else None)
    cfg = TrainConfig(
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=max(1, args.steps // 10),
        accumulate=1,
        amp_dtype=amp,
        log_every=1,
    )
    records = train_steps(
        model,
        random_batches(args.batch_size, args.seq_len, vocab, device),
        steps=args.steps,
        out_dir=args.out,
        log_path=args.log or (args.out / "smoke.jsonl"),
        cfg=cfg,
        on_step=lambda row: print(json.dumps({"step": row["step"], "loss": row["loss"], "tokens": row["tokens"]})),
    )
    save_checkpoint(model, args.out / "final")

    # Reload through the same public entry point inference uses.
    reloaded = load_model(str(args.out / "final"), backend="torch" if args.tiny or device != "cuda" else args.backend, device=device, dtype=dtype)
    reloaded.eval()
    with torch.no_grad():
        again = model_logits(reloaded, probe.cpu() if next(reloaded.parameters()).device.type == "cpu" else probe)
    print(json.dumps({"reload_ok": True, "reload_logits_shape": list(again.shape), "steps": len(records)}))


if __name__ == "__main__":
    main()
