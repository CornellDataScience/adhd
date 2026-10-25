# adhd

CDS research project: how well do **Mamba-3** language models handle code and codebase
questions under tight compute, compared with similarly sized Transformers? The end product
is a RAG assistant for CDS docs and repos (ingest → pgvector retrieval → Mamba-3 /
Transformer `/generate` → answer + citations).

This repo currently contains only the **Mamba-3 model setup**. The RAG components
(`ingest/`, `index/`, `retrieval/`, `inference/`, `prompting_eval/`, `app/`) come next.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and torch 2.14.1 are pinned so
laptops and cluster nodes match.

**Laptop (macOS / CPU / any OS):**

```bash
uv sync
```

**Cluster (Linux + NVIDIA GPU):** this also builds the official `mamba_ssm` kernels from
`state-spaces/mamba` (pinned commit) and uses CUDA 12.6 torch wheels:

```bash
uv sync --extra cuda
```

## Two backends, one interface

| Backend | Where | What |
|---|---|---|
| `official` | Linux + CUDA | `mamba_ssm.MambaLMHeadModel` with Triton/TileLang/CuTe kernels. SISO and MIMO. Use for benchmarks and serving. |
| `torch` | anywhere (CPU, Apple MPS, CUDA) | [`mamba3/mamba3_torch.py`](mamba3/mamba3_torch.py), a pure-PyTorch port that loads the same checkpoints. **SISO only.** Use for development and debugging. |

`load_model(..., backend="auto")` picks `official` when `mamba_ssm` and a GPU are
available, otherwise `torch`. Both expose the same `generate(input_ids, max_length, top_k, top_p, temperature)`.

```python
from transformers import AutoTokenizer
from mamba3 import TOKENIZER_ID, load_model

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = load_model("siso-443m")  # or a full HF id / local checkpoint dir
ids = tokenizer("def fibonacci(n):", return_tensors="pt").input_ids.to(next(model.parameters()).device)
print(tokenizer.decode(model.generate(ids, max_length=ids.shape[1] + 48)[0]))
```

CLI:

```bash
uv run python -m mamba3.generate --model siso-187m --prompt "Cornell Data Science is"
```

Checkpoints (`state-spaces/*` on the HF Hub, pretrained on 100B FineWeb-Edu tokens, 2k context,
Llama-3.1 tokenizer): `siso-187m`, `siso-443m`, `siso-893m`, `siso-1.5b`, and
`mimo-187m`, `mimo-444m`, `mimo-894m`, `mimo-1.5b` (MIMO needs the `official` backend).

## Tests

```bash
uv run pytest
```

```bash
uv run pytest -m slow
```

The default suite checks the PyTorch port against the upstream reference recurrence
(`tests/_mamba3_reference.py`, copied from state-spaces/mamba): chunked prefill, decoding
step by step, continuing from a saved state, and the backward pass. The `slow` suite
downloads `siso-187m` and checks perplexity and that prefill and decoding agree.

## Notes

- **Measured on an M5 MacBook (torch backend, MPS, fp32):** 187M runs at about 22 tok/s and
  443M at about 28 tok/s (greedy decoding). Perplexity on Python source is about 102 for
  187M and 77 for 443M, which is weak because the checkpoints never saw code in
  pretraining. Expect to need the design doc's domain-adaptive training step.
- **The `cuda` extra has not been tested on the cluster yet.** If the cluster has Blackwell
  GPUs (sm_100+) or drivers for CUDA 13, change the `pytorch-cu126` index in
  `pyproject.toml` to `cu130`.
- To bump the `mamba_ssm` commit, update `rev` in `[tool.uv.sources]` and the copied
  `requires-dist` in `[[tool.uv.dependency-metadata]]`. That metadata lets `uv lock`
  run on macOS without building the package.
- Set `HF_TOKEN` to avoid Hub rate limits.
