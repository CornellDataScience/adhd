"""Decide which repo files get chunked, and turn notebooks into plain Python.

Repos can add a `.ragignore` (gitignore syntax) at their root to exclude more paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import nbformat
import pathspec

MAX_FILE_BYTES = 512 * 1024
MAX_DATA_FILE_BYTES = 64 * 1024  # JSON/YAML/XML above this is data, not config
MAX_LINE_CHARS = 1000
MAX_MEAN_LINE_CHARS = 200

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
    "*.sqlite", "*.sqlite3", "*.db", "*.dta", "*.sav", "*.rds", "*.RData",
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


@dataclass
class Prepared:
    source: bytes
    language: str | None = None  # overrides detection by extension


@dataclass
class Skipped:
    reason: str


def load_ignore(repo_dir: Path) -> pathspec.GitIgnoreSpec:
    lines = list(DEFAULT_IGNORE)
    ragignore = repo_dir / ".ragignore"
    if ragignore.is_file():
        lines += ragignore.read_text().splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


def prepare_file(path: str, raw: bytes, ignore: pathspec.GitIgnoreSpec) -> Prepared | Skipped:
    if ignore.match_file(path):
        return Skipped("ignored path")
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
    return Prepared(raw)


def _looks_minified(text: str) -> bool:
    lines = text.splitlines() or [""]
    return max(map(len, lines)) > MAX_LINE_CHARS and len(text) / len(lines) > MAX_MEAN_LINE_CHARS


def _prepare_notebook(raw: bytes) -> Prepared | Skipped:
    """Source cells only, in jupytext percent format; outputs are dropped."""
    try:
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
