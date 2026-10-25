import json
import subprocess
from collections import Counter

import pytest

from ingest.chunker import chunk_repo
from ingest.filters import MAX_FILE_BYTES, Prepared, Skipped, load_ignore, prepare_file


@pytest.fixture
def ignore(tmp_path):
    return load_ignore(tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "uv.lock", "web/package-lock.json", "data/trace.csv", "model.pt", "img/logo.png",
        "web/node_modules/react/index.js", "pkg/__pycache__/a.pyc", "nb/.ipynb_checkpoints/x.ipynb",
        "static/app.min.js", ".env", ".gitignore", "proto/api_pb2.py",
    ],
)
def test_default_ignores(path, ignore):
    assert prepare_file(path, b"x = 1\n", ignore) == Skipped("ignored path")


@pytest.mark.parametrize("path", ["src/main.py", "README.md", "Dockerfile", "jobs/run.sbatch", "env/gym_env.py", "LICENSE"])
def test_source_files_pass(path, ignore):
    assert prepare_file(path, b"x = 1\n", ignore) == Prepared(b"x = 1\n")


def test_content_checks(ignore):
    assert prepare_file("a.py", b"\x00\x01", ignore) == Skipped("binary")
    assert prepare_file("a.py", b"\n  \n", ignore) == Skipped("empty")
    assert prepare_file("a.py", b"x" * (MAX_FILE_BYTES + 1), ignore) == Skipped("too large")
    assert prepare_file("a.json", b"[" + b"1," * 40_000 + b"1]", ignore) == Skipped("data file")
    assert prepare_file("a.js", b"var a=1;" * 500, ignore) == Skipped("minified")
    assert prepare_file("config.json", b'{"a": 1}\n', ignore) == Prepared(b'{"a": 1}\n')


def test_ragignore(tmp_path):
    (tmp_path / ".ragignore").write_text("experiments/\n")
    ignore = load_ignore(tmp_path)
    assert prepare_file("experiments/run.py", b"x = 1\n", ignore) == Skipped("ignored path")
    assert prepare_file("src/run.py", b"x = 1\n", ignore) == Prepared(b"x = 1\n")


def notebook(cells, language="python"):
    return json.dumps({
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"language_info": {"name": language}},
        "cells": [
            {"cell_type": t, "id": str(i), "metadata": {}, "source": src,
             **({"outputs": [{"output_type": "stream", "name": "stdout", "text": "OUTPUT"}], "execution_count": 1} if t == "code" else {})}
            for i, (t, src) in enumerate(cells)
        ],
    }).encode()


def test_notebook_becomes_percent_script(ignore):
    raw = notebook([("markdown", "# Title\nSome text"), ("code", "%matplotlib inline\n!pip install x\nimport os"), ("code", "")])
    prepared = prepare_file("nb/analysis.ipynb", raw, ignore)
    assert prepared.language == "python"
    text = prepared.source.decode()
    assert text == "# %% [markdown]\n# # Title\n# Some text\n\n# %%\n# %matplotlib inline\n# !pip install x\nimport os\n"
    assert "OUTPUT" not in text


def test_bad_notebooks(ignore):
    assert prepare_file("a.ipynb", b"not json", ignore) == Skipped("invalid notebook")
    assert prepare_file("a.ipynb", notebook([("code", "  ")]), ignore) == Skipped("empty")


def test_chunk_repo_applies_filters(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "nb.ipynb").write_bytes(notebook([("code", "x = 1")]))
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")

    skipped = Counter()
    records = list(chunk_repo(tmp_path, "org/repo", lambda t: len(t.split()), skipped, scan_secrets=False))
    assert sorted(r["path"] for r in records) == ["nb.ipynb", "src/app.py"]
    assert skipped == {"ignored path": 1}
    nb = next(r for r in records if r["path"] == "nb.ipynb")
    assert nb["language"] == "python" and "#L" not in nb["source_url"]
