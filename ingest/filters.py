"""Decide which repo files get chunked, and turn notebooks into plain Python.

Repos can add a `.ragignore` (gitignore syntax) at their root to exclude more paths.
"""

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import nbformat
import pathspec
import tree_sitter_language_pack as tslp

MAX_FILE_BYTES = 512 * 1024
MAX_DATA_FILE_BYTES = 64 * 1024  # JSON/YAML/XML above this is data, not config
MAX_LINE_CHARS = 1000
MAX_MEAN_LINE_CHARS = 200
MIN_TABULAR_ROWS = 10

DEFAULT_IGNORE = [
    # dependencies, build output, caches
    "node_modules/", "vendor/", "third_party/", "site-packages/", ".venv/", "venv/",
    "dist/", "build/", "target/", ".next/", ".nuxt/", "coverage/", "htmlcov/",
    "__pycache__/", ".ipynb_checkpoints/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
    ".tox/", ".gradle/", ".idea/", ".vscode/", ".git/",
    # lockfiles
    "*.lock", "package-lock.json", "pnpm-lock.yaml", "npm-shrinkwrap.json", "go.sum",
    # data
    "*.csv", "*.tsv", "*.jsonl", "*.ndjson", "*.parquet", "*.arrow", "*.feather", "*.avro",
    "*.xls", "*.xlsx", "*.pkl", "*.pickle", "*.npy", "*.npz", "*.h5", "*.hdf5", "*.mat",
    "*.sqlite", "*.sqlite3", "*.db", "*.dta", "*.sav", "*.rds", "*.rdata",
    "*.vw", "*.dat", "*.data", "*.log", "*.out",
    # model weights
    "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.onnx", "*.bin", "*.gguf", "*.tflite", "*.joblib",
    # media, documents, archives, fonts
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg", "*.webp", "*.tif", "*.tiff",
    "*.mp3", "*.wav", "*.mp4", "*.mov", "*.avi", "*.webm",
    "*.pdf", "*.doc", "*.docx", "*.ppt", "*.pptx", "*.key",
    "*.zip", "*.tar", "*.gz", "*.tgz", "*.bz2", "*.xz", "*.7z", "*.rar",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot",
    # generated or minified
    "*.min.js", "*.min.css", "*.map", "*_pb2.py", "*_pb2_grpc.py",
    # environment files may hold secrets
    ".env", ".env.*",
    # tool config with nothing to answer questions from
    ".gitignore", ".gitattributes", ".gitmodules", ".dockerignore", ".python-version",
    ".editorconfig", ".prettierrc*", ".eslintignore", ".npmrc", ".nvmrc",
]
DATA_SUFFIXES = {".json", ".yaml", ".yml", ".xml", ".geojson"}
# Inside these folders only source code and docs are kept (e.g. data/loader.py stays, data/train.json doesn't).
DATA_DIRS = {
    "data", "dataset", "datasets", "raw", "processed", "interim", "output", "outputs", "results",
    "logs", "runs", "checkpoints", "artifacts", "wandb", "mlruns", "lightning_logs", "predictions",
}
DOC_SUFFIXES = {".md", ".markdown", ".rst", ".ipynb"}
NON_CODE_LANGUAGES = {"csv", "tsv", "json", "yaml", "xml", "toml", "ini", "vimdoc", "requirements"}


@dataclass
class Prepared:
    source: bytes
    language: str | None = None  # overrides detection by extension


@dataclass
class Skipped:
    reason: str


class IgnoreRules:
    """Default patterns match case-insensitively (`*.CSV`); a repo's .ragignore matches as written."""

    def __init__(self, ragignore_lines: list[str]):
        self.defaults = pathspec.GitIgnoreSpec.from_lines(DEFAULT_IGNORE)
        self.repo = pathspec.GitIgnoreSpec.from_lines(ragignore_lines)

    def match_file(self, path: str) -> bool:
        return self.defaults.match_file(path.lower()) or self.repo.match_file(path)


def load_ignore(repo_dir: Path) -> IgnoreRules:
    ragignore = repo_dir / ".ragignore"
    return IgnoreRules(ragignore.read_text().splitlines() if ragignore.is_file() else [])


def is_code_or_docs(path: str) -> bool:
    if PurePosixPath(path).suffix.lower() in DOC_SUFFIXES:
        return True
    language = tslp.detect_language_from_path(path)
    return language is not None and language not in NON_CODE_LANGUAGES


def prepare_file(path: str, raw: bytes, ignore: IgnoreRules) -> Prepared | Skipped:
    if ignore.match_file(path):
        return Skipped("ignored path")
    in_data_dir = any(part.lower() in DATA_DIRS for part in PurePosixPath(path).parts[:-1])
    if in_data_dir and not is_code_or_docs(path):
        return Skipped("data directory")
    if b"\0" in raw[:8192]:
        return Skipped("binary")
    if PurePosixPath(path).suffix == ".ipynb":
        return _prepare_notebook(raw)
    if len(raw) > MAX_FILE_BYTES:
        return Skipped("too large")
    if PurePosixPath(path).suffix.lower() in DATA_SUFFIXES and len(raw) > MAX_DATA_FILE_BYTES:
        return Skipped("data file")
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return Skipped("empty")
    if _looks_minified(text):
        return Skipped("minified")
    if not is_code_or_docs(path) and _looks_tabular(text):
        return Skipped("tabular data")
    return Prepared(raw)


def _looks_minified(text: str) -> bool:
    lines = text.splitlines() or [""]
    return max(map(len, lines)) > MAX_LINE_CHARS and len(text) / len(lines) > MAX_MEAN_LINE_CHARS


def _looks_tabular(text: str) -> bool:
    """Most rows share the same number (>= 2) of commas, tabs or semicolons."""
    rows = [line for line in text.splitlines() if line.strip()][:200]
    if len(rows) < MIN_TABULAR_ROWS:
        return False
    for delimiter in (",", "\t", ";"):
        count, rows_with_count = Counter(row.count(delimiter) for row in rows).most_common(1)[0]
        if count >= 2 and rows_with_count / len(rows) >= 0.8:
            return True
    return False


def _prepare_notebook(raw: bytes) -> Prepared | Skipped:
    """Source cells only, in jupytext percent format; outputs are dropped."""
    try:
        with warnings.catch_warnings():  # old notebooks lack cell ids; harmless here
            warnings.simplefilter("ignore")
            nb = nbformat.reads(raw.decode("utf-8"), as_version=4)
    except (UnicodeDecodeError, json.JSONDecodeError, nbformat.reader.NotJSONError, nbformat.validator.ValidationError):
        return Skipped("invalid notebook")
    language = nb.metadata.get("language_info", {}).get("name") or nb.metadata.get("kernelspec", {}).get("language") or "python"
    comment = "#" if language in {"python", "r", "julia", "ruby", "shell", "bash"} else "//"
    parts = []
    for cell in nb.cells:
        source = cell.source.rstrip()
        if not source:
            continue
        if cell.cell_type == "markdown":
            body = "\n".join(f"{comment} {line}".rstrip() for line in source.splitlines())
            parts.append(f"{comment} %% [markdown]\n{body}")
        elif cell.cell_type == "code":
            # IPython magics and shell escapes aren't valid code; keep them as comments.
            body = "\n".join(f"{comment} {line}" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())
            parts.append(f"{comment} %%\n{body}")
    if not parts:
        return Skipped("empty")
    source = ("\n\n".join(parts) + "\n").encode()
    if len(source) > MAX_FILE_BYTES:
        return Skipped("too large")
    return Prepared(source, language=language)
