"""Split files into chunks of at most MAX_TOKENS along function, class and heading boundaries.

Small neighbouring pieces are merged until a chunk reaches MIN_TOKENS; pieces over
MAX_TOKENS are split into their children (methods, paragraphs), then into lines.
Output records follow contracts/chunk.schema.json.

    uv run python -m ingest.chunker path/to/repo --repo CornellDataScience/name --out chunks.jsonl
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

import tree_sitter_language_pack as tslp

from ingest.filters import Skipped, load_ignore, prepare_file

MIN_TOKENS = 200
MAX_TOKENS = 400

DOC_LANGUAGES = {"markdown", "rst", "text"}
DEFINITION_RE = re.compile(r"function|class|method|struct|interface|enum|trait|impl|module|mod_item|type|namespace|macro")
LEADING_NODE_RE = re.compile(r"comment|attribute|decorator|annotation")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

TokenCounter = Callable[[str], int]


@lru_cache(maxsize=1)
def llama_token_counter() -> TokenCounter:
    from transformers import AutoTokenizer

    from mamba3 import TOKENIZER_ID

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


@dataclass
class Unit:
    """Lines [start, end] (0-based, inclusive) of a file, plus what they define."""

    start: int
    end: int
    symbols: list[str] = field(default_factory=list)
    node: object | None = None  # tree-sitter node, when the unit is one syntax node
    prefix: list[str] = field(default_factory=list)  # enclosing definitions, for qualified names


def detect_language(path: str) -> str:
    if PurePosixPath(path).suffix.lower() in {".md", ".markdown"}:
        return "markdown"
    if PurePosixPath(path).suffix.lower() == ".rst":
        return "rst"
    return tslp.detect_language_from_path(path) or "text"


def chunk_file(
    repo: str,
    path: str,
    source: bytes,
    commit_sha: str,
    count_tokens: TokenCounter,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
    language: str | None = None,
) -> list[dict]:
    text = source.decode("utf-8", errors="replace")
    lines = text.split("\n")
    language = language or detect_language(path)
    # Line numbers of converted notebooks don't match the .ipynb on GitHub, so link the file.
    line_anchors = PurePosixPath(path).suffix != ".ipynb"

    def render(unit: Unit) -> str:
        symbol = ", ".join(dict.fromkeys(unit.symbols))
        header = f"# {repo} {path}:{unit.start + 1}-{unit.end + 1}" + (f" {symbol}" if symbol else "")
        return header + "\n" + "\n".join(lines[unit.start : unit.end + 1])

    cache: dict[tuple, int] = {}

    def tokens(unit: Unit) -> int:
        key = (unit.start, unit.end, tuple(unit.symbols))
        if key not in cache:
            cache[key] = count_tokens(render(unit))
        return cache[key]

    if language == "markdown":
        units = _markdown_units(lines)
    elif language in DOC_LANGUAGES:
        units = _paragraph_units(lines, 0, len(lines) - 1, [])
    else:
        units = _code_units(language, source, lines)

    chunks = _pack(units, lines, tokens, min_tokens, max_tokens)
    file_hash = hashlib.sha256(source).hexdigest()
    name = PurePosixPath(path).name.lower()
    kind = "readme" if name.startswith("readme") else "doc" if language in DOC_LANGUAGES else "code"

    records = []
    for unit in chunks:
        body = render(unit)
        if not "\n".join(lines[unit.start : unit.end + 1]).strip():
            continue
        start, end = unit.start + 1, unit.end + 1
        records.append(
            {
                "id": hashlib.sha256(f"{repo}\0{path}\0{start}\0{body}".encode()).hexdigest(),
                "repo": repo,
                "path": path,
                "start_line": start,
                "end_line": end,
                "symbol": ", ".join(dict.fromkeys(unit.symbols)) or None,
                "language": language,
                "kind": kind,
                "commit_sha": commit_sha,
                "source_url": f"https://github.com/{repo}/blob/{commit_sha}/{path}" + (f"#L{start}-L{end}" if line_anchors else ""),
                "file_hash": file_hash,
                "token_count": tokens(unit),
                "text": body,
            }
        )
    return records


def _pack(units, lines, tokens, min_tokens, max_tokens) -> list[Unit]:
    out: list[Unit] = []
    current: Unit | None = None
    queue = deque(units)
    while queue:
        unit = queue.popleft()
        if tokens(unit) > max_tokens and unit.end > unit.start:  # one huge line stays whole
            queue.extendleft(reversed(_split(unit, lines)))
            continue
        if current is not None:
            merged = Unit(current.start, unit.end, current.symbols + unit.symbols)
            if tokens(current) < min_tokens and tokens(merged) <= max_tokens:
                current = merged
                continue
            out.append(current)
        current = unit
    if current:
        out.append(current)
    # A short tail is better joined to its predecessor when it fits.
    if len(out) >= 2 and tokens(out[-1]) < min_tokens:
        merged = Unit(out[-2].start, out[-1].end, out[-2].symbols + out[-1].symbols)
        if tokens(merged) <= max_tokens:
            out[-2:] = [merged]
    return out


def _split(unit: Unit, lines: list[str]) -> list[Unit]:
    """Children of an oversized unit: definitions in its body, else paragraphs, else lines."""
    if unit.node is not None:
        children = _body_units(unit)
        if children:
            return children
    paragraphs = _paragraph_units(lines, unit.start, unit.end, unit.symbols)
    if len(paragraphs) > 1:
        return paragraphs
    return [Unit(i, i, unit.symbols) for i in range(unit.start, unit.end + 1)]


# ---- code -------------------------------------------------------------------


def _code_units(language: str, source: bytes, lines: list[str]) -> list[Unit]:
    try:
        tree = tslp.get_parser(language).parse(source)
    except Exception:  # unknown grammar or failed download: fall back to plain text
        return _paragraph_units(lines, 0, len(lines) - 1, [])
    return _node_units(tree.root_node.named_children, prefix=[])


def _node_units(nodes, prefix: list[str]) -> list[Unit]:
    units: list[Unit] = []
    pending: Unit | None = None  # comments and attributes, kept with the next node
    for node in nodes:
        name = _definition_name(node)
        symbols = [".".join(prefix + [name])] if name else []
        end = node.end_point[0] - (node.end_point[1] == 0 and node.end_point[0] > node.start_point[0])
        unit = Unit(node.start_point[0], end, symbols, node, prefix + ([name] if name else []))
        if pending and pending.end + 1 >= unit.start:
            unit.start = pending.start
        elif pending:
            units.append(pending)
        pending = None
        if LEADING_NODE_RE.search(node.type):
            pending = Unit(unit.start, unit.end)
            continue
        units.append(unit)
    if pending:
        units.append(pending)
    return units


def _definition_name(node) -> str | None:
    for wrapper_field in ("definition", "declaration"):  # Python decorators, JS/TS exports
        inner = node.child_by_field_name(wrapper_field)
        if inner is not None:
            return _definition_name(inner)
    if not DEFINITION_RE.search(node.type):
        return None
    name = node.child_by_field_name("name") or node.child_by_field_name("type")  # Rust impl blocks
    return name.text.decode("utf-8", errors="replace") if name is not None else None


def _body_units(unit: Unit) -> list[Unit]:
    node = unit.node
    for wrapper_field in ("definition", "declaration"):
        node = node.child_by_field_name(wrapper_field) or node
    body = node.child_by_field_name("body")
    if body is None or len(body.named_children) < 2:
        return []
    children = _node_units(body.named_children, unit.prefix)
    for child in children:
        child.symbols = child.symbols or unit.symbols
    if children[0].start > unit.start:  # signature, decorators, opening brace
        children.insert(0, Unit(unit.start, children[0].start - 1, unit.symbols))
    if children[-1].end < unit.end:  # closing brace
        children[-1].end = unit.end
    return children


# ---- docs -------------------------------------------------------------------


def _markdown_units(lines: list[str]) -> list[Unit]:
    units: list[Unit] = []
    path: list[tuple[int, str]] = []
    start, in_fence = 0, False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if not match:
            continue
        if i > start:
            units.append(Unit(start, i - 1, [" > ".join(t for _, t in path)] if path else []))
        level = len(match.group(1))
        path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, match.group(2))]
        start = i
    units.append(Unit(start, len(lines) - 1, [" > ".join(t for _, t in path)] if path else []))
    return units


def _paragraph_units(lines: list[str], start: int, end: int, symbols: list[str]) -> list[Unit]:
    units, block_start = [], None
    for i in range(start, end + 1):
        if lines[i].strip():
            if block_start is None:
                block_start = i
        elif block_start is not None:
            units.append(Unit(block_start, i - 1, symbols))
            block_start = None
    if block_start is not None:
        units.append(Unit(block_start, end, symbols))
    return units or [Unit(start, end, symbols)]


# ---- repos ------------------------------------------------------------------


def chunk_repo(repo_dir: Path, repo: str, count_tokens: TokenCounter, skipped: Counter | None = None) -> Iterator[dict]:
    """Chunk every tracked file that passes ingest.filters; counts skip reasons into `skipped`."""
    git = lambda *args: subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True).stdout  # noqa: E731
    commit_sha = git("rev-parse", "HEAD").strip()
    ignore = load_ignore(repo_dir)
    for path in git("ls-files", "-z").split("\0"):
        file = repo_dir / path
        if not path or not file.is_file() or file.is_symlink():
            continue
        prepared = prepare_file(path, file.read_bytes(), ignore)
        if isinstance(prepared, Skipped):
            if skipped is not None:
                skipped[prepared.reason] += 1
            continue
        yield from chunk_file(repo, path, prepared.source, commit_sha, count_tokens, language=prepared.language)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("--repo", required=True, help="owner/name on GitHub")
    parser.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    args = parser.parse_args()
    count_tokens = llama_token_counter()
    n, skipped = 0, Counter()
    with args.out.open("w") as f:
        for record in chunk_repo(args.repo_dir, args.repo, count_tokens, skipped):
            f.write(json.dumps(record) + "\n")
            n += 1
    print(f"Wrote {n} chunks to {args.out}; skipped files: {dict(skipped) or 'none'}")


if __name__ == "__main__":
    main()
