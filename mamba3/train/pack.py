"""Pack public code into 2048-token blocks with the Llama 3.1 tokenizer.

    uv run python -m mamba3.train.pack --local-dir path/to/py --out data/packed --max-tokens 2000000
    uv run python -m mamba3.train.pack --dataset bigcode/starcoderdata --data-dir python --out data/packed
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from transformers import AutoTokenizer

from mamba3 import TOKENIZER_ID
from mamba3.train.data import write_packed

DEFAULT_SEQ_LEN = 2048
CODE_SUFFIXES = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".md"}


def _repo_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    parts = rel.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else path.name


def iter_local_documents(root: Path, holdout_repos: set[str]) -> Iterator[tuple[str, str, str]]:
    """Yield ``(repo, path, text)`` for text files under ``root``."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        repo = _repo_from_path(path, root)
        if repo in holdout_repos:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        yield repo, str(path.relative_to(root)), text


def iter_dataset_documents(
    dataset: str,
    data_dir: str | None,
    holdout_repos: set[str],
    max_documents: int | None,
) -> Iterator[tuple[str, str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Packing from Hugging Face needs the datasets package. "
            "Install with: uv add datasets   or   pip install datasets"
        ) from exc

    kwargs = {"split": "train", "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    ds = load_dataset(dataset, **kwargs)
    count = 0
    for row in ds:
        repo = row.get("max_stars_repo_name") or row.get("repo_name") or row.get("repository_name") or ""
        path = row.get("max_stars_repo_path") or row.get("path") or row.get("file_name") or ""
        text = row.get("content") or row.get("text") or ""
        if not text or not isinstance(text, str):
            continue
        if repo in holdout_repos:
            continue
        yield str(repo), str(path), text
        count += 1
        if max_documents is not None and count >= max_documents:
            break


def pack_documents(
    documents: Iterator[tuple[str, str, str]],
    tokenizer,
    *,
    seq_len: int,
    max_tokens: int | None,
    drop_short_final: bool = True,
) -> tuple[list[list[int]], list[dict], int]:
    """Concatenate documents with EOS between them and cut into ``seq_len`` blocks."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")

    buffer: list[int] = []
    blocks: list[list[int]] = []
    sources: list[dict] = []
    tokens_emitted = 0

    for repo, path, text in documents:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        piece = list(ids) + [eos]
        sources.append({"repo": repo, "path": path, "bytes": len(text.encode("utf-8")), "tokens": len(ids)})
        buffer.extend(piece)
        while len(buffer) >= seq_len:
            blocks.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            tokens_emitted += seq_len
            if max_tokens is not None and tokens_emitted >= max_tokens:
                return blocks, sources, tokens_emitted

    if buffer and not drop_short_final:
        # Only full blocks are written; short finals are dropped by default.
        pass
    return blocks, sources, tokens_emitted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Directory for tokens.int32.bin and manifest.json")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after this many packed tokens")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--holdout-repo", action="append", default=[], help="Repo name to exclude; repeatable")
    parser.add_argument("--local-dir", type=Path, help="Directory of source files to pack")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset id (e.g. bigcode/starcoderdata)")
    parser.add_argument("--data-dir", default="python", help="Dataset data_dir for language slice")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--keep-short-final", action="store_true", help="Record only; short finals are still dropped")
    args = parser.parse_args()

    if bool(args.local_dir) == bool(args.dataset):
        raise SystemExit("Pass exactly one of --local-dir or --dataset")

    holdout = set(args.holdout_repo)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.local_dir:
        documents = iter_local_documents(args.local_dir, holdout)
    else:
        documents = iter_dataset_documents(args.dataset, args.data_dir, holdout, args.max_documents)

    drop_short = not args.keep_short_final
    blocks, sources, tokens_emitted = pack_documents(
        documents,
        tokenizer,
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        drop_short_final=drop_short,
    )
    if not blocks:
        raise SystemExit("No full blocks packed. Lower --seq-len or add more documents.")

    write_packed(
        blocks,
        args.out,
        seq_len=args.seq_len,
        sources=sources,
        holdout_repos=sorted(holdout),
        drop_short_final=drop_short,
    )
    print(json.dumps({"out": str(args.out), "n_blocks": len(blocks), "tokens": tokens_emitted, "sources": len(sources)}))


if __name__ == "__main__":
    main()
