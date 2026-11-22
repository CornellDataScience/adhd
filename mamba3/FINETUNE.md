# Fine-tuning plan

Continued pretraining of Mamba-3 on code, then a small supervised pass so it answers from retrieved chunks. The embedder and the reranker stay as they are.

The checkpoints (`state-spaces/mamba3-siso-*`) were trained on FineWeb-Edu web text, 2k context, Llama 3.1 tokenizer. Python perplexity on the laptop is about 102 for `siso-187m` and 77 for `siso-443m`. That is the number this plan has to move. Retrieval does not fix a model that has never modeled code.

No trainer exists. `mamba3/generate.py` and `inference/prototype_server.py` only generate. `from_pretrained` returns the model in eval mode, and `generate` runs under `inference_mode`, so neither can be the training entry point.

## Where it runs

Train on a rented Linux NVIDIA GPU (Runpod RTX 4090 Secure Cloud is the current choice), official `mamba_ssm` kernels. Do not start a long Pod until the smoke CLI has been run once on that machine.

```bash
uv sync --extra cuda
```

`load_model(..., backend="auto")` picks those kernels when CUDA and `mamba_ssm` are both present. If the host is CUDA 13, change the `pytorch-cu126` index in `pyproject.toml` to `cu130` before the first job.

The laptop PyTorch port stays for unit tests and for reading checkpoints. Packing can run on a laptop. The GPU job only reads packed token files.

Start with `siso-187m`. MIMO checkpoints need the official backend and are out of this plan. `siso-443m` is the second run, after 187M has moved the Python perplexity. `siso-893m` and `siso-1.5b` wait until the recipe is known.

## Implemented modules

These exist under `mamba3/train/` (and `prompting_eval/prompt.py`). Full weights only. No LoRA.

| Module | Command |
|---|---|
| Pack | `uv run python -m mamba3.train.pack --local-dir … --out …` or `--dataset bigcode/starcoderdata --data-dir python` |
| Smoke | `uv run python -m mamba3.train.smoke --tiny --steps 5 --out /tmp/smoke` (wiring). On GPU: `--model siso-187m --steps 10 --seq-len 128 --out /workspace/smoke` |
| CPT | `uv run python -m mamba3.train.cpt --model siso-187m --data …/packed --out …/ckpts --token-budget …` |
| SFT | `uv run python -m mamba3.train.sft --model …/ckpts/last --data sft.jsonl --out …/sft` |
| PPL | `uv run python -m mamba3.train.ppl --model … --data …/packed` or `--text-file …` |

Unit + integration (no Hub download): `uv run pytest tests/test_train.py tests/test_train_integration.py`.  
The integration suite runs the tiny smoke CLI, pack → CPT → reload → ppl, and citation SFT masking.

## LoRA

Do not use LoRA for the code-adaptation stage.

That stage has to change how the model predicts the next token on code. In `mamba3/mamba3_torch.py` the mixer is not a stack of attention linears. Each layer's `in_proj` emits the SSM inputs (`z`, `x`, `B`, `C`, `dt`, `A`, trap, RoPE angles), and the recurrence itself is carried by `dt_bias`, `B_bias`, `C_bias`, `D`, and the `B`/`C` norms, then `out_proj`. LoRA on `in_proj` and `out_proj` leaves those SSM parameters at their web-text values. The domain shift is exactly what those parameters have to absorb.

`siso-187m` is small enough to fine-tune in full. Weights in bf16 plus AdamW moments are a few gigabytes. Activations at sequence length 2048 dominate memory, and the first smoke batch on the cluster is what measures that. Full fine-tuning of 443M is still a single-GPU job if that smoke batch fits. If it does not, gradient checkpointing comes before LoRA.

LoRA is an option only for the later citation pass, and only if the code-adapted checkpoint must stay frozen so adapters can be swapped. At 187M that pass is also small enough to run in full. The embeddings and `lm_head` are tied (`tie_embeddings` defaults to true, and `from_pretrained` re-ties them). A LoRA wrapper must not copy those two tensors apart.

## What gets trained

| Model | This plan |
|---|---|
| Mamba-3 SISO, starting at `siso-187m` | Continued pretraining on code, then supervised answers with citations |
| `Qwen/Qwen3-Embedding-0.6B` | Leave it. It is already a code embedder, and the CDS corpus is small. |
| `Qwen/Qwen3-Reranker-0.6B` | Leave it, for the same reason. |
| Tokenizer `unsloth/Llama-3.1-8B` | Leave it. A new token would resize the tied embedding. |

Hold the eval repos out of both training sets before the first run. If the model trains on the files the questions ask about, a higher score is partly memorization. Public code is the continued-pretraining corpus. CDS facts still come from `retrieve()`.

## Modules

Built under `mamba3/train/`. They load the model and write a directory `load_model` can read back.

`mamba3/train/pack.py`
Tokenize with `TOKENIZER_ID` and pack documents into length 2048, the checkpoint context. Emit a memmap or a directory of `.bin` files of `int32` token ids, plus a small manifest (source repo, byte count, holdout flag). No padding inside a packed sequence. A short final sequence is dropped or kept and masked, and the choice is recorded in the manifest.

`mamba3/train/loss.py`
Next-token cross-entropy. Logits at position `t` predict token `t + 1`. Positions that are padding or that belong to the prompt, during the supervised pass, contribute nothing. The torch port's `forward` returns `(logits, states)`. The official `MambaLMHeadModel` return value has to be confirmed on the cluster before this function assumes that tuple.

`mamba3/train/checkpoint.py`
Write `config.json` and `pytorch_model.bin` in the layout `_load_checkpoint` already reads: a local directory with those two files. Keep the embedding tie on save and on load. `load_model` on that directory is the reload test.

`mamba3/train/loop.py`
Shared step: `model.train()`, bf16 autocast on CUDA, AdamW, gradient clip, accumulation, a JSONL log of loss and tokens seen, and a call into `checkpoint.py`. This module does not know whether the batch is raw code or a citation example.

`mamba3/train/cpt.py`
CLI for continued pretraining. Reads a packed corpus, runs `loop.py`, stops on a token budget or on a held-out perplexity plateau.

`mamba3/train/sft.py`
CLI for the citation pass. Same loop. The batch carries a mask so the loss is only on the answer tokens.

`mamba3/train/ppl.py`
Held-out Python perplexity in the same setup as the README number (torch or official, recorded in the log). This is the comparison against 102 and 77. It also runs a few held-out questions through `retrieve()` and `generate` once a prompt builder exists.

`prompting_eval/prompt.py`
One function: retrieve response in, string out. Chunks numbered `[1]`, `[2]`, … in rank order, then the question, with an instruction to cite those numbers or abstain. `sft.py` and the inference server both call it. The package is a README today, so this is a real module, not a note.

Leave `retrieval/`, `index/embed.py`, and `inference/prototype_server.py` alone except for a later switch of the model path to the saved directory. The prototype server generates from a raw prompt and does not call `retrieve()`.

## Steps

### 0. Baseline before any gradient

Load `samples/chunks.jsonl`, run `index/embed.py`, call `retrieve()`, build a prompt by hand for one question, and generate with the untouched `siso-187m`. Record the answer and the Python perplexity. Every later checkpoint is compared to that pair. The supervised data cannot be built until `retrieve()` returns real chunks.

### 1. Prove the cluster can train one step

On the cluster:

1. `uv sync --extra cuda`.
2. Load `siso-187m` with `backend="official"`.
3. One forward on a short token batch. Record the logits shape. It has to be `(batch, seq, vocab)`.
4. One backward on a mean of those logits. Gradients finite.
5. Save with `checkpoint.py` and load the directory through `load_model`.

If step 2 fails, fix the CUDA extra (including the `cu130` index if the GPU requires it) before writing a corpus. A long run on the torch port is not a substitute.

### 2. Freeze the split

Write down the eval repos and the question list first. Those repos never enter `pack.py`. The 100-chunk sample and whatever the 50-question set covers count as held out until someone explicitly moves a repo to train.

### 3. Build the continued-pretraining corpus

Public code with a license the project can train on. Python is the first slice, because that is the perplexity already measured. Other languages the CDS repos use can follow in a second slice.

`pack.py` then:

- Tokenizes with the Llama 3.1 tokenizer already used by `mamba3` and by the chunker.
- Concatenates documents with the tokenizer's end-of-sequence id between them.
- Cuts the stream into blocks of 2048.
- Writes the holdout manifest so a training shard cannot contain an eval repo.

Do not add tokens to the vocabulary. Do not change the context length on the first run. The checkpoint was trained at 2k.

### 4. Smoke run

Ten optimizer steps on `siso-187m`, batch as large as the GPU allows, sequence 2048. The loss should fall across those steps. The saved directory must load and generate. Delete the smoke checkpoint. It is not a result.

Starting settings, to be changed if the smoke loss diverges or sits still. These are not a tuned recipe:

- AdamW, learning rate `2e-5`, weight decay `0.1`, betas `0.9` and `0.95`.
- Warmup over the first 1% of the planned steps, cosine decay to `2e-6`.
- Gradient clip `1.0`.
- bf16 parameters and autocast. Adam moments in fp32.
- Effective batch of about 256 sequences of 2048 tokens if memory allows, reached by accumulation. A smaller batch is acceptable for the smoke run.

### 5. Continued pretraining

Run `cpt.py` until held-out Python perplexity stops falling for a predeclared window, or until a token budget chosen before the run is spent. Log tokens seen, loss, and learning rate. Save a checkpoint on that same window and keep the one with the best held-out perplexity.

`ppl.py` on that checkpoint is the gate. The comparison is the README baseline: about 102 for 187M and, later, 77 for 443M. A small drop is not a reason to start the citation pass.

### 6. Same questions as the baseline

Point `load_model` at the saved directory. Ask the same questions used in step 0, with the same retrieved chunks. The only change is the weights. Keep the answers beside the baseline answers.

### 7. Supervised citation pass

Build examples only after step 6. Each row is:

- question
- the `retrieve()` response for that question
- an answer that cites `[1]` and `[2]` in rank order, or abstains when the chunks do not contain the answer

`prompting_eval/prompt.py` renders the input. `sft.py` masks every prompt token out of the loss and trains on the answer tokens. The data set is small and written or checked by hand. It is not a scrape of the web.

Start from the step-5 checkpoint, at a lower learning rate than continued pretraining (`2e-6` is the starting value). A few epochs over that small set. Stop when the model cites real ranks on the held-out questions and still abstains on a question whose chunks are unrelated. Run `ppl.py` again. If Python perplexity regresses to the baseline, the citation pass overfit and the step-5 checkpoint remains the one to ship.

Full weights, not LoRA, unless step 5 has to stay bit-identical and the citation behavior has to be an adapter on top. That constraint is not required for the first experiment.

### 8. The comparison the project is for

Repeat steps 3 through 7 on a Transformer of similar size, same tokenizer family if the checkpoint allows it, same packed code, same citation set, same held-out questions. The Mamba run and the Transformer run differ by architecture, not by data.

### 9. Scale

If 187M clears the perplexity gate and the citation check, repeat steps 4 through 7 for `siso-443m`. Do not start 893M or 1.5B until 443M has a recorded perplexity and a recorded set of answers.

## What this plan does not build

A trainer on the laptop. A LoRA CPT run. A fine-tune of the Qwen embedder or reranker. A new tokenizer. A context length above 2048. MIMO. Training on the eval repos. A change to BM25, fusion, or the reranker.
