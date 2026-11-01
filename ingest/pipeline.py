"""Crawl the org, then chunk every synced repo into one chunks.jsonl.

    uv run python -m ingest.pipeline --root data/repos --out data/chunks.jsonl --only FTM-Snake slurm-aware
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ingest import crawler
from ingest.chunker import chunk_repo, llama_token_counter

CHUNKABLE = {"ok", "stale"}


def run(root: Path, out: Path, repos: list[crawler.RepoInfo], count_tokens, jobs=8, prune=True, scan_secrets=True) -> dict:
    entries = crawler.crawl(root, repos, jobs=jobs, prune=prune)
    listed = {r.full_name for r in repos}
    skipped, redacted, chunks = Counter(), [], Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w") as f:
        for name in sorted(listed):
            entry = entries[name]
            if entry.status not in CHUNKABLE:
                continue
            for record in chunk_repo(root / entry.path, name, count_tokens, skipped, redacted, scan_secrets):
                f.write(json.dumps(record) + "\n")
                chunks[name] += 1
    tmp.replace(out)  # a failed run never leaves a half-written file
    return {"entries": entries, "chunks": chunks, "skipped": skipped, "redacted": redacted}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/repos"), help="Where clones and manifest.json live")
    parser.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--config", type=Path, default=crawler.CONFIG_PATH)
    parser.add_argument("--only", nargs="+", default=(), help="Crawl just these repos (no pruning)")
    parser.add_argument("--limit", type=int, help="Crawl the first N repos (no pruning)")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel clones")
    parser.add_argument("--no-secret-scan", action="store_true", help="Skip gitleaks (never for real ingestion)")
    args = parser.parse_args()

    config = crawler.load_config(args.config)
    listing = crawler.list_org_repos(config["org"], os.environ.get("GITHUB_TOKEN"))
    repos = crawler.select_repos(listing, config.get("exclude", []), args.only, args.limit)
    if args.only and len(repos) < len(set(args.only)):
        found = {r.full_name.split("/")[-1] for r in repos}
        print(f"Not found or not indexable: {sorted(set(args.only) - found)}")
    partial = bool(args.only or args.limit)

    result = run(args.root, args.out, repos, llama_token_counter(), args.jobs, prune=not partial, scan_secrets=not args.no_secret_scan)

    statuses = Counter(e.status for e in result["entries"].values())
    print(f"Repos: {dict(statuses)}")
    for e in sorted(result["entries"].values(), key=lambda e: e.full_name):
        if e.status in {"failed", "stale", "skipped"}:
            print(f"  {e.status}: {e.full_name}: {e.error}")
        elif e.status == "removed":
            print(f"  removed: {e.full_name}")
    print(f"Wrote {sum(result['chunks'].values())} chunks from {len(result['chunks'])} repos to {args.out}")
    print(f"Skipped files: {dict(result['skipped']) or 'none'}")
    if args.no_secret_scan:
        print("WARNING: secret scan disabled")
    for finding in result["redacted"]:
        print(f"Redacted {finding.rule} at {finding.path}:{finding.line}")


if __name__ == "__main__":
    main()
