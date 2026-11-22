"""Training helpers without Hub downloads or a GPU."""

import json
from pathlib import Path

import pytest
import torch

from mamba3.mamba3_torch import Mamba3LMHeadModel
from mamba3.train.checkpoint import load_trainable, model_config_dict, save_checkpoint
from mamba3.train.data import PackedCorpus, write_packed
from mamba3.train.logits import model_logits
from mamba3.train.loss import next_token_loss
from mamba3.train.loop import StepBatch, TrainConfig, train_steps
from mamba3.train.pack import pack_documents
from prompting_eval.prompt import build_prompt

TINY = {
    "d_model": 64,
    "d_intermediate": 128,
    "n_layer": 2,
    "vocab_size": 128,
    "ssm_cfg": {"layer": "Mamba3", "d_state": 32, "headdim": 16, "chunk_size": 16},
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True,
}


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        # Stable fake ids from character ordinals, capped into vocab.
        ids = [min(127, max(2, ord(ch) % 120)) for ch in text[:40]]
        return {"input_ids": ids or [2]}


def test_next_token_loss_and_mask():
    logits = torch.randn(2, 5, 10)
    ids = torch.randint(0, 10, (2, 5))
    loss, n = next_token_loss(logits, ids)
    assert n == 8
    assert torch.isfinite(loss)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 3:] = True
    masked_loss, masked_n = next_token_loss(logits, ids, mask)
    assert masked_n == 4
    assert torch.isfinite(masked_loss)


def test_pack_and_packed_corpus(tmp_path: Path):
    docs = iter(
        [
            ("org/a", "a.py", "def one():\n    return 1\n"),
            ("org/b", "b.py", "def two():\n    return 2\n" * 20),
        ]
    )
    blocks, sources, tokens = pack_documents(docs, FakeTokenizer(), seq_len=16, max_tokens=64)
    assert blocks
    assert all(len(block) == 16 for block in blocks)
    assert sources and tokens >= 16
    write_packed(blocks, tmp_path, seq_len=16, sources=sources, holdout_repos=["org/hold"])
    corpus = PackedCorpus(tmp_path)
    assert len(corpus) == len(blocks)
    assert corpus[0].shape == (16,)
    assert corpus.batch([0]).shape == (1, 16)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["holdout_repos"] == ["org/hold"]


def test_checkpoint_roundtrip_and_train_step(tmp_path: Path):
    model = Mamba3LMHeadModel(TINY, device="cpu", dtype=torch.float32)
    model.train()
    ids = torch.randint(0, 128, (2, 24))
    logits = model_logits(model, ids)
    assert logits.shape == (2, 24, 128)
    loss, _ = next_token_loss(logits, ids)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    out = save_checkpoint(model, tmp_path / "ckpt")
    assert (out / "config.json").is_file()
    assert (out / "pytorch_model.bin").is_file()
    assert "vocab_size" in model_config_dict(model)

    reloaded = load_trainable(str(out), backend="torch", device="cpu", dtype=torch.float32)
    assert next(reloaded.parameters()).requires_grad
    with torch.no_grad():
        again = model_logits(reloaded, ids)
    assert again.shape == logits.shape


def test_train_steps_loss_moves(tmp_path: Path):
    model = Mamba3LMHeadModel(TINY, device="cpu", dtype=torch.float32)
    model.train()
    vocab = 128

    def batches():
        while True:
            yield StepBatch(input_ids=torch.randint(0, vocab, (2, 32)))

    records = train_steps(
        model,
        batches(),
        steps=5,
        out_dir=tmp_path / "run",
        log_path=tmp_path / "run.jsonl",
        cfg=TrainConfig(lr=1e-3, total_steps=5, warmup_steps=1, amp_dtype=None, accumulate=1),
    )
    assert len(records) == 5
    assert (tmp_path / "run" / "last" / "pytorch_model.bin").is_file()
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in records)


def test_build_prompt_numbers_sources():
    response = {
        "question": "How does load_model pick a backend?",
        "results": [
            {
                "rank": 1,
                "score": 0.9,
                "chunk": {
                    "repo": "CornellDataScience/adhd",
                    "path": "mamba3/__init__.py",
                    "symbol": "load_model",
                    "text": "def load_model(...):\n    ...\n",
                },
            }
        ],
    }
    text = build_prompt(response)
    assert "Question: How does load_model pick a backend?" in text
    assert "[1] CornellDataScience/adhd mamba3/__init__.py (load_model)" in text
    assert text.strip().endswith("Answer:")
