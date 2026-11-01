"""Keep shallow clones of the org's repos under a root folder, recorded in root/manifest.json."""

import json
import os
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("repos.toml")
GIT_TIMEOUT_SECONDS = 600


@dataclass
class RepoInfo:
    full_name: str
    clone_url: str
    default_branch: str
    fork: bool = False
    archived: bool = False
    empty: bool = False


@dataclass
class ManifestEntry:
    full_name: str
    path: str  # relative to the root
    default_branch: str
    commit_sha: str | None
    status: str  # ok | stale | skipped | failed | removed
    synced_at: str
    error: str | None = None


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def list_org_repos(org: str, token: str | None = None) -> list[RepoInfo]:
    from github import Auth, Github

    gh = Github(auth=Auth.Token(token)) if token else Github()
    return [
        RepoInfo(r.full_name, r.clone_url, r.default_branch, r.fork, r.archived, r.size == 0)
        for r in gh.get_organization(org).get_repos(type="public")
    ]


def select_repos(repos: list[RepoInfo], exclude=(), only=(), limit: int | None = None) -> list[RepoInfo]:
    names = lambda r: {r.full_name, r.full_name.split("/")[-1]}  # noqa: E731
    selected = [
        r for r in sorted(repos, key=lambda r: r.full_name.lower())
        if not (r.fork or r.archived or r.empty) and not names(r) & set(exclude) and (not only or names(r) & set(only))
    ]
    return selected[:limit] if limit else selected


def crawl(root: Path, repos: list[RepoInfo], jobs: int = 8, prune: bool = True) -> dict[str, ManifestEntry]:
    """Clone or update each repo; with prune, delete clones of repos no longer listed."""
    root.mkdir(parents=True, exist_ok=True)
    previous = read_manifest(root)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        entries = {e.full_name: e for e in pool.map(lambda r: sync_repo(root, r), repos)}
    for name, old in previous.items():
        if name in entries:
            continue
        if prune and old.status != "removed":
            shutil.rmtree(root / old.path, ignore_errors=True)
            entries[name] = ManifestEntry(name, old.path, old.default_branch, None, "removed", _now())
        elif not prune:
            entries[name] = old
    write_manifest(root, entries)
    return entries


def sync_repo(root: Path, repo: RepoInfo) -> ManifestEntry:
    rel = repo.full_name
    dest = root / rel
    entry = lambda status, sha=None, error=None: ManifestEntry(  # noqa: E731
        repo.full_name, rel, repo.default_branch, sha, status, _now(), error
    )
    if not (dest / ".git").is_dir():
        shutil.rmtree(dest, ignore_errors=True)  # leftovers of an interrupted clone
        try:
            _git(None, "clone", "--quiet", "--depth", "1", "--single-branch", "--branch", repo.default_branch, repo.clone_url, str(dest))
        except RuntimeError as e:
            shutil.rmtree(dest, ignore_errors=True)
            return entry("failed", error=str(e))
        return entry("ok", _git(dest, "rev-parse", "HEAD"))

    if _git(dest, "status", "--porcelain"):
        return entry("skipped", _git(dest, "rev-parse", "HEAD"), "local changes in the clone")
    try:
        _git(dest, "fetch", "--quiet", "--depth", "1", "origin", repo.default_branch)
        _git(dest, "reset", "--quiet", "--hard", "FETCH_HEAD")
    except RuntimeError as e:
        return entry("stale", _git(dest, "rev-parse", "HEAD"), str(e))  # keep indexing the last good copy
    return entry("ok", _git(dest, "rev-parse", "HEAD"))


def read_manifest(root: Path) -> dict[str, ManifestEntry]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    return {name: ManifestEntry(**e) for name, e in json.loads(path.read_text())["repos"].items()}


def write_manifest(root: Path, entries: dict[str, ManifestEntry]):
    data = {"synced_at": _now(), "repos": {name: asdict(entries[name]) for name in sorted(entries)}}
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(root / "manifest.json")


def _git(cwd: Path | None, *args: str) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # fail instead of prompting for credentials
    try:
        result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {args[0]} timed out") from None
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode}")
    return result.stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
