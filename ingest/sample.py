"""Build a small, deterministic sample of chunks across several repos.

    uv run python -m ingest.sample path/to/repo:CornellDataScience/name ... --n 100 --out samples/chunks.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ingest.chunker import chunk_repo, llama_token_counter


def sample_chunks(records: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to n records, one per file per round, so many files are represented."""
    rng = random.Random(seed)
    by_file = defaultdict(list)
    for record in records:
        by_file[record["path"]].append(record)
    files = sorted(by_file)
    rng.shuffle(files)
    for path in files:
        rng.shuffle(by_file[path])
    picked = []
    while len(picked) < n and any(by_file.values()):
        for path in files:
            if by_file[path] and len(picked) < n:
                picked.append(by_file[path].pop())
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repos", nargs="+", help="repo_dir:owner/name")
    parser.add_argument("--n", type=int, default=100, help="Total chunks, split evenly across repos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("samples/chunks.jsonl"))
    args = parser.parse_args()

    count_tokens = llama_token_counter()
    repos = [spec.rsplit(":", 1) for spec in args.repos]
    picked = []
    for i, (repo_dir, name) in enumerate(repos):
        quota = (args.n - len(picked)) // (len(repos) - i)
        picked += sample_chunks(list(chunk_repo(Path(repo_dir), name, count_tokens)), quota, args.seed)

    picked.sort(key=lambda r: (r["repo"], r["path"], r["start_line"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in picked:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(picked)} chunks from {len(repos)} repos to {args.out}")


if __name__ == "__main__":
    main()
