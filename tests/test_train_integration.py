"""Integration tests for pack → train → checkpoint → reload (no Hub download)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mamba3 import load_model
from mamba3.mamba3_torch import Mamba3LMHeadModel
from mamba3.train.checkpoint import save_checkpoint
from mamba3.train.data import PackedCorpus, write_packed
from mamba3.train.logits import model_logits
from mamba3.train.loop import StepBatch, TrainConfig, train_steps
from mamba3.train.pack import pack_documents
from mamba3.train.ppl import perplexity_on_corpus
from mamba3.train.sft import encode_example, load_examples
from prompting_eval.prompt import build_prompt

REPO = Path(__file__).resolve().parents[1]

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
        ids = [min(127, max(2, (ord(ch) % 120))) for ch in text]
        if len(ids) < 8:
            ids = ids + [2] * (8 - len(ids))
        return {"input_ids": ids}


def test_smoke_cli_tiny_end_to_end(tmp_path: Path):
    out = tmp_path / "smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mamba3.train.smoke",
            "--tiny",
            "--steps",
            "3",
            "--seq-len",
            "32",
            "--batch-size",
            "2",
            "--out",
            str(out),
            "--log",
            str(out / "smoke.jsonl"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "reload_ok" in proc.stdout
    assert (out / "final" / "pytorch_model.bin").is_file()
    rows = [json.loads(line) for line in (out / "smoke.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert all("loss" in row for row in rows)


def test_pack_cpt_ppl_reload_pipeline(tmp_path: Path):
    src = tmp_path / "src" / "org" / "demo"
    src.mkdir(parents=True)
    (src / "a.py").write_text("def hello():\n    return 1\n" * 40)
    (src / "b.py").write_text("def world():\n    return 2\n" * 40)

    from mamba3.train.pack import iter_local_documents

    docs = iter_local_documents(tmp_path / "src", holdout_repos={"org/hold"})
    blocks, sources, tokens = pack_documents(
        docs,
        FakeTokenizer(),
        seq_len=32,
        max_tokens=512,
    )
    assert blocks and sources and tokens >= 32
    packed = write_packed(blocks, tmp_path / "packed", seq_len=32, sources=sources, holdout_repos=["org/hold"])
    corpus = PackedCorpus(packed)
    assert len(corpus) >= 1

    model = Mamba3LMHeadModel(TINY, device="cpu", dtype=torch.float32)
    model.train()

    def batches():
        order = list(range(len(corpus)))
        while True:
            for start in range(0, len(order), 2):
                chunk = order[start : start + 2]
                if len(chunk) < 2:
                    chunk = [order[0], order[0]]
                yield StepBatch(input_ids=corpus.batch(chunk))

    ckpt = tmp_path / "cpt"
    records = train_steps(
        model,
        batches(),
        steps=4,
        out_dir=ckpt,
        log_path=ckpt / "train.jsonl",
        cfg=TrainConfig(lr=1e-3, total_steps=4, warmup_steps=1, amp_dtype=None, accumulate=1, save_every=2),
    )
    assert len(records) == 4
    assert (ckpt / "last" / "config.json").is_file()
    assert (ckpt / "step-2" / "pytorch_model.bin").is_file()

    reloaded = load_model(str(ckpt / "last"), backend="torch", device="cpu", dtype=torch.float32)
    reloaded.eval()
    with torch.no_grad():
        logits = model_logits(reloaded, corpus.batch([0]))
    assert logits.shape == (1, 32, 128)

    stats = perplexity_on_corpus(reloaded, corpus, max_blocks=min(4, len(corpus)))
    assert stats["tokens"] > 0
    assert stats["perplexity"] > 1.0
    assert (tmp_path / "packed" / "manifest.json").is_file()


def test_sft_prompt_mask_and_train(tmp_path: Path):
    retrieve_response = {
        "question": "How does load_model pick a backend?",
        "results": [
            {
                "rank": 1,
                "score": 0.9,
                "chunk": {
                    "repo": "CornellDataScience/adhd",
                    "path": "mamba3/__init__.py",
                    "symbol": "load_model",
                    "text": "def load_model(model_id, backend=\"auto\"):\n    ...\n",
                },
            }
        ],
    }
    prompt = build_prompt(retrieve_response)
    answer = "It uses backend=\"auto\" to pick official or torch. [1]"
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(
        json.dumps(
            {
                "question": retrieve_response["question"],
                "retrieve_response": retrieve_response,
                "answer": answer,
            }
        )
        + "\n"
        + json.dumps({"prompt": "Q?\nAnswer:\n", "answer": "Because [1]."})
        + "\n"
    )
    examples = load_examples(sft_path)
    assert len(examples) == 2
    assert "[1]" in examples[0]["prompt"]

    tok = FakeTokenizer()
    batch = encode_example(tok, examples[0]["prompt"], examples[0]["answer"], max_length=256)
    assert batch.input_ids.shape[0] == 1
    assert batch.loss_mask is not None
    assert bool(batch.loss_mask[0, 0].item()) is False or batch.loss_mask.any()
    # At least one answer token is supervised.
    assert int(batch.loss_mask.sum().item()) >= 1

    model = Mamba3LMHeadModel(TINY, device="cpu", dtype=torch.float32)
    model.train()

    def batches():
        while True:
            # Re-encode into tiny vocab space by clamping ids.
            ids = batch.input_ids.clamp(max=127)
            mask = batch.loss_mask
            yield StepBatch(input_ids=ids, loss_mask=mask)

    records = train_steps(
        model,
        batches(),
        steps=3,
        out_dir=tmp_path / "sft-ckpt",
        cfg=TrainConfig(lr=1e-3, total_steps=3, warmup_steps=0, amp_dtype=None),
    )
    assert len(records) == 3
    save_checkpoint(model, tmp_path / "sft-final")
    assert (tmp_path / "sft-final" / "pytorch_model.bin").is_file()


@pytest.mark.slow
def test_smoke_siso_187m_one_step_if_checkpoint_cached(tmp_path: Path):
    """Optional Hub path. Skipped in default CI (`not slow`)."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache("state-spaces/mamba3-siso-187m", "pytorch_model.bin")
    if cached is None:
        pytest.skip("siso-187m not cached; run with network once before enabling")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mamba3.train.smoke",
            "--model",
            "siso-187m",
            "--backend",
            "torch",
            "--steps",
            "1",
            "--seq-len",
            "32",
            "--out",
            str(tmp_path / "187m"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "reload_ok" in proc.stdout
