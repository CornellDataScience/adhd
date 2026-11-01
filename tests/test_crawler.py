import json
import subprocess

import pytest

from ingest import crawler, pipeline
from ingest.crawler import RepoInfo, crawl, read_manifest, select_repos


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_remote(tmp_path, name, files, branch="main"):
    path = tmp_path / "remotes" / name
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", branch)
    commit(path, files)
    return RepoInfo(f"org/{name}", path.as_uri(), branch)


def commit(path, files):
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    git(path, "add", ".")
    git(path, "commit", "-qm", "change")
    return git(path, "rev-parse", "HEAD")


def head(remote: RepoInfo):
    return git(remote.clone_url.removeprefix("file://"), "rev-parse", "HEAD")


@pytest.fixture
def root(tmp_path):
    return tmp_path / "root"


def test_select_repos():
    repos = [
        RepoInfo("org/b", "", "main"),
        RepoInfo("org/a", "", "main"),
        RepoInfo("org/fork", "", "main", fork=True),
        RepoInfo("org/old", "", "main", archived=True),
        RepoInfo("org/empty", "", "main", empty=True),
        RepoInfo("org/skip", "", "main"),
    ]
    assert [r.full_name for r in select_repos(repos, exclude=["skip"])] == ["org/a", "org/b"]
    assert [r.full_name for r in select_repos(repos, only=["b"])] == ["org/b"]
    assert [r.full_name for r in select_repos(repos, only=["fork"])] == []
    assert [r.full_name for r in select_repos(repos, limit=1)] == ["org/a"]


def test_clone_then_update(tmp_path, root):
    remote = make_remote(tmp_path, "app", {"main.py": "x = 1\n"})
    entries = crawl(root, [remote])
    assert entries["org/app"].status == "ok"
    assert entries["org/app"].commit_sha == head(remote)
    assert git(root / "org/app", "rev-parse", "--is-shallow-repository") == "true"

    new_sha = commit(tmp_path / "remotes/app", {"main.py": "x = 2\n"})
    entries = crawl(root, [remote])
    assert entries["org/app"].commit_sha == new_sha
    assert (root / "org/app/main.py").read_text() == "x = 2\n"
    assert read_manifest(root)["org/app"].commit_sha == new_sha


def test_non_main_default_branch(tmp_path, root):
    remote = make_remote(tmp_path, "legacy", {"a.py": "x = 1\n"}, branch="master")
    assert crawl(root, [remote])["org/legacy"].status == "ok"


def test_local_changes_are_left_alone(tmp_path, root):
    remote = make_remote(tmp_path, "app", {"main.py": "x = 1\n"})
    old_sha = crawl(root, [remote])["org/app"].commit_sha
    (root / "org/app/main.py").write_text("edited locally\n")
    commit(tmp_path / "remotes/app", {"main.py": "x = 2\n"})
    entry = crawl(root, [remote])["org/app"]
    assert entry.status == "skipped" and entry.commit_sha == old_sha
    assert (root / "org/app/main.py").read_text() == "edited locally\n"


def test_failures_are_isolated(tmp_path, root):
    good = make_remote(tmp_path, "good", {"a.py": "x = 1\n"})
    bad = RepoInfo("org/missing", (tmp_path / "nope").as_uri(), "main")
    entries = crawl(root, [good, bad])
    assert entries["org/good"].status == "ok"
    assert entries["org/missing"].status == "failed" and entries["org/missing"].error
    assert not (root / "org/missing").exists()


def test_fetch_failure_keeps_last_good_copy(tmp_path, root):
    remote = make_remote(tmp_path, "app", {"main.py": "x = 1\n"})
    sha = crawl(root, [remote])["org/app"].commit_sha
    moved = RepoInfo("org/app", (tmp_path / "gone").as_uri(), "main")
    git(root / "org/app", "remote", "set-url", "origin", moved.clone_url)
    entry = crawl(root, [moved])["org/app"]
    assert entry.status == "stale" and entry.commit_sha == sha


def test_prune_removes_only_repos_the_crawler_cloned(tmp_path, root):
    a = make_remote(tmp_path, "a", {"a.py": "x = 1\n"})
    b = make_remote(tmp_path, "b", {"b.py": "x = 1\n"})
    crawl(root, [a, b])
    (root / "someone-elses-dir").mkdir()

    entries = crawl(root, [a], prune=False)  # partial runs (--only/--limit) never delete
    assert (root / "org/b").exists() and entries["org/b"].status == "ok"

    entries = crawl(root, [a])
    assert entries["org/b"].status == "removed"
    assert not (root / "org/b").exists()
    assert (root / "someone-elses-dir").exists()
    assert "org/b" not in crawl(root, [a])  # reported once, then dropped


def test_pipeline_writes_combined_chunks(tmp_path, root):
    a = make_remote(tmp_path, "a", {"src/app.py": "def main():\n    return 1\n", "data/x.csv": "a,b\n"})
    b = make_remote(tmp_path, "b", {"README.md": "# B\n\nHello.\n"})
    out = tmp_path / "chunks.jsonl"
    result = pipeline.run(root, out, [a, b], lambda t: len(t.split()), jobs=2, scan_secrets=False)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {(r["repo"], r["path"]) for r in records} == {("org/a", "src/app.py"), ("org/b", "README.md")}
    assert {r["commit_sha"] for r in records if r["repo"] == "org/a"} == {head(a)}
    assert result["chunks"] == {"org/a": 1, "org/b": 1}
    assert not out.with_name(out.name + ".tmp").exists()


def test_pipeline_skips_repos_it_could_not_sync(tmp_path, root):
    good = make_remote(tmp_path, "good", {"a.py": "x = 1\n"})
    bad = RepoInfo("org/missing", (tmp_path / "nope").as_uri(), "main")
    result = pipeline.run(root, tmp_path / "out.jsonl", [good, bad], lambda t: len(t.split()), scan_secrets=False)
    assert result["chunks"] == {"org/good": 1}


def test_default_config_is_valid():
    config = crawler.load_config()
    assert config["org"] == "CornellDataScience"
    assert isinstance(config["exclude"], list)
