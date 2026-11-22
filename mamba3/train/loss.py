"""Next-token cross-entropy for continued pretraining and citation SFT."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy of token ``t`` predicting ``t + 1``.

    ``logits`` is ``(batch, seq, vocab)``. ``input_ids`` is ``(batch, seq)``.
    Optional ``loss_mask`` is ``(batch, seq)`` booleans over the *label* positions
    (aligned with ``input_ids[:, 1:]``). False positions are ignored. Prompt tokens
    in the citation pass are masked out this way.

    Returns ``(mean_loss, n_tokens)``. ``n_tokens`` is 0 and the loss is 0 when
    every position is masked.
    """
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and input_ids {tuple(input_ids.shape)} must share batch and seq"
        )
    if input_ids.shape[1] < 2:
        raise ValueError("sequence length must be at least 2 for next-token loss")

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_labels = shift_labels.reshape(-1)

    if loss_mask is None:
        loss = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")
        return loss, int(flat_labels.numel())

    if loss_mask.shape != input_ids.shape:
        raise ValueError(
            f"loss_mask {tuple(loss_mask.shape)} must match input_ids {tuple(input_ids.shape)}"
        )
    # Label at position i (predicting token i+1) uses mask of the label token index i+1.
    label_mask = loss_mask[:, 1:].reshape(-1).bool()
    n_tokens = int(label_mask.sum().item())
    if n_tokens == 0:
        return flat_logits.sum() * 0.0, 0
    loss = F.cross_entropy(flat_logits.float()[label_mask], flat_labels[label_mask], reduction="mean")
    return loss, n_tokens
