"""Shared AdamW training step. Full weights, no LoRA."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from mamba3.train.checkpoint import save_checkpoint
from mamba3.train.logits import model_logits
from mamba3.train.loss import next_token_loss


@dataclass
class TrainConfig:
    lr: float = 2e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 0
    min_lr: float = 2e-6
    total_steps: int = 1000
    accumulate: int = 1
    log_every: int = 1
    save_every: int = 0
    amp_dtype: str | None = "bf16"  # bf16 | fp16 | None


def _amp_dtype(name: str | None, device: torch.device) -> torch.dtype | None:
    if name is None or device.type != "cuda":
        return None
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown amp_dtype {name!r}")


def _learning_rate(step: int, cfg: TrainConfig) -> float:
    if cfg.total_steps <= 0:
        return cfg.lr
    if step < cfg.warmup_steps:
        return cfg.lr * float(step + 1) / max(1, cfg.warmup_steps)
    if cfg.total_steps <= cfg.warmup_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


@dataclass
class StepBatch:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor | None = None


def train_steps(
    model,
    batches: Iterator[StepBatch],
    *,
    steps: int,
    out_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    cfg: TrainConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run ``steps`` optimizer updates. Returns the per-log records.

    ``batches`` yields ``StepBatch`` values. Gradient accumulation uses
    ``cfg.accumulate`` micro-batches per update. Checkpoints go to ``out_dir``
    when ``save_every`` is positive or at the end if ``out_dir`` is set.
    """
    cfg = cfg or TrainConfig(total_steps=steps)
    device = next(model.parameters()).device
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

    amp = _amp_dtype(cfg.amp_dtype, device)
    use_scaler = amp == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    log_file = Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    tokens_seen = 0
    micro = 0
    step = 0
    running_loss = 0.0
    running_tokens = 0

    while step < steps:
        try:
            batch = next(batches)
        except StopIteration:
            break
        input_ids = batch.input_ids.to(device)
        loss_mask = None if batch.loss_mask is None else batch.loss_mask.to(device)

        if amp is None:
            logits = model_logits(model, input_ids)
            loss, n_tokens = next_token_loss(logits, input_ids, loss_mask)
        else:
            with torch.autocast(device_type="cuda", dtype=amp):
                logits = model_logits(model, input_ids)
                loss, n_tokens = next_token_loss(logits, input_ids, loss_mask)

        if n_tokens == 0:
            continue

        scaled = loss / cfg.accumulate
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        running_loss += float(loss.detach().item()) * n_tokens
        running_tokens += n_tokens
        micro += 1
        if micro < cfg.accumulate:
            continue

        lr = _learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += running_tokens
        mean_loss = running_loss / max(1, running_tokens)
        record = {
            "step": step,
            "loss": mean_loss,
            "lr": lr,
            "tokens": tokens_seen,
            "grad_norm": float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm),
            "time": time.time(),
        }
        if step % cfg.log_every == 0:
            records.append(record)
            if log_file:
                with log_file.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            if on_step:
                on_step(record)

        if out_dir and cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            save_checkpoint(model, Path(out_dir) / f"step-{step + 1}")

        running_loss = 0.0
        running_tokens = 0
        micro = 0
        step += 1

    if out_dir:
        save_checkpoint(model, Path(out_dir) / "last")
    return records
