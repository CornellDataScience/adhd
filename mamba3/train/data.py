"""Packed int32 token blocks for continued pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

TOKENS_NAME = "tokens.int32.bin"
MANIFEST_NAME = "manifest.json"


class PackedCorpus:
    """Memmap of shape ``(n_blocks, seq_len)`` int32 token ids."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / MANIFEST_NAME).read_text())
        self.seq_len = int(self.manifest["seq_len"])
        self.n_blocks = int(self.manifest["n_blocks"])
        path = self.root / TOKENS_NAME
        expected = self.n_blocks * self.seq_len
        self._mmap = np.memmap(path, dtype=np.int32, mode="r", shape=(expected,))
        if self._mmap.size != expected:
            raise ValueError(f"{path} has {self._mmap.size} ids, expected {expected}")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.n_blocks:
            raise IndexError(index)
        start = index * self.seq_len
        block = np.array(self._mmap[start : start + self.seq_len], dtype=np.int64)
        return torch.from_numpy(block)

    def batch(self, indices: list[int]) -> torch.Tensor:
        return torch.stack([self[i] for i in indices], dim=0)


def write_packed(
    token_blocks: list[list[int]] | np.ndarray,
    out_dir: str | Path,
    *,
    seq_len: int,
    sources: list[dict] | None = None,
    holdout_repos: list[str] | None = None,
    drop_short_final: bool = True,
) -> Path:
    """Write ``tokens.int32.bin`` and ``manifest.json``. Each block must be ``seq_len`` long."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(token_blocks, np.ndarray):
        array = np.asarray(token_blocks, dtype=np.int32)
        if array.ndim != 2 or array.shape[1] != seq_len:
            raise ValueError(f"expected (n, {seq_len}) array, got {array.shape}")
        blocks = array
    else:
        for block in token_blocks:
            if len(block) != seq_len:
                raise ValueError(f"block length {len(block)} != seq_len {seq_len}")
        blocks = np.asarray(token_blocks, dtype=np.int32)
        if blocks.ndim != 2:
            blocks = blocks.reshape(-1, seq_len)

    flat = blocks.reshape(-1)
    path = out / TOKENS_NAME
    mmap = np.memmap(path, dtype=np.int32, mode="w+", shape=(flat.size,))
    mmap[:] = flat
    mmap.flush()
    del mmap

    manifest = {
        "seq_len": seq_len,
        "n_blocks": int(blocks.shape[0]),
        "dtype": "int32",
        "tokens_file": TOKENS_NAME,
        "drop_short_final": drop_short_final,
        "holdout_repos": list(holdout_repos or []),
        "sources": list(sources or []),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return out
