import json
import random
import shutil
import string
import subprocess

import pytest

from ingest import secret_scan
from ingest.chunker import chunk_repo
from ingest.secret_scan import Finding, GitleaksNotFound, redact

needs_gitleaks = pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")


def fake_github_token():
    # Built at runtime so this file doesn't itself contain a token gitleaks would flag.
    rng = random.Random(0)
    return "ghp" + "_" + "".join(rng.choices(string.ascii_letters + string.digits, k=36))


def test_redact_keeps_line_numbers():
    raw = b"a = 1\nkey = 'line1\nline2'\nb = key\nagain = 'line1\nline2'\n"
    out = redact(raw, [Finding("f.py", "private-key", 2, "line1\nline2")])
    assert out == b"a = 1\nkey = '[REDACTED:private-key]\n'\nb = key\nagain = '[REDACTED:private-key]\n'\n"
    assert out.count(b"\n") == raw.count(b"\n")


def test_missing_gitleaks_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("GITLEAKS", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GitleaksNotFound):
        secret_scan.scan(tmp_path)


def make_repo(path, token):
    (path / "src").mkdir()
    (path / "src" / "app.py").write_text(f'import os\n\nTOKEN = "{token}"\n\n\ndef f():\n    return TOKEN\n')
    nb = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "id": "a", "metadata": {}, "source": f'key = "{token}"', "outputs": [], "execution_count": 1}],
    }
    (path / "nb.ipynb").write_text(json.dumps(nb))
    git = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")


@needs_gitleaks
def test_chunk_repo_redacts_secrets(tmp_path):
    token = fake_github_token()
    make_repo(tmp_path, token)
    redacted = []
    records = list(chunk_repo(tmp_path, "org/repo", lambda t: len(t.split()), redacted=redacted))
    assert sorted((f.path, f.rule) for f in redacted) == [("nb.ipynb", "github-pat"), ("src/app.py", "github-pat")]
    text = "\n".join(r["text"] for r in records)
    assert token not in text
    assert text.count("[REDACTED:github-pat]") == 2


@needs_gitleaks
def test_chunk_repo_refuses_unredacted_secret(tmp_path, monkeypatch):
    make_repo(tmp_path, fake_github_token())
    monkeypatch.setattr(secret_scan, "redact", lambda raw, findings: raw)
    with pytest.raises(RuntimeError, match="survived redaction"):
        list(chunk_repo(tmp_path, "org/repo", lambda t: len(t.split())))
