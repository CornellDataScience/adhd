"""Save and reload directories in the layout ``load_model`` already reads."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mamba3 import load_model


def model_config_dict(model) -> dict:
    """Return a JSON-serializable config matching ``config.json`` on Hub checkpoints."""
    cfg = model.config
    if isinstance(cfg, dict):
        return json.loads(json.dumps(cfg))
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    raw = {key: value for key, value in vars(cfg).items() if not key.startswith("_")}
    return json.loads(json.dumps(raw, default=str))


def save_checkpoint(model, path: str | Path) -> Path:
    """Write ``config.json`` and ``pytorch_model.bin``. Keeps tied embedding weights once."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    config = model_config_dict(model)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # state_dict already shares storage for tied lm_head / embedding when tied.
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    return out


def load_trainable(
    model_id: str,
    *,
    backend: str = "auto",
    device: str | None = None,
    dtype: torch.dtype | None = None,
):
    """Load a Hub id or local checkpoint directory in train mode (not eval)."""
    model = load_model(model_id, backend=backend, device=device, dtype=dtype)
    model.train()
    return model
