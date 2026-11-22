"""Normalize Mamba-3 forward outputs to a logits tensor."""

from __future__ import annotations

import torch


def model_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return ``(batch, seq, vocab)`` logits from the torch port or official kernels.

    The torch port returns ``(logits, states)``. Official ``MambaLMHeadModel`` may
    return the same tuple or an object with a ``logits`` attribute. Confirm the
    shape on the first smoke step on the GPU host.
    """
    out = model(input_ids)
    if isinstance(out, tuple):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    if logits.ndim != 3:
        raise ValueError(f"expected logits of shape (batch, seq, vocab), got {tuple(logits.shape)}")
    return logits
