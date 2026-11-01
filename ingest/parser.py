"""Parse existing chunks without splitting, merging, or changing their text.

    uv run python -m ingest.parser chunks.jsonl --out parsed_chunks.jsonl

Input: one JSON object per line, with text, path, and start_line. The existing
ingest.chunker output is accepted directly, including its source header.
Output: {"chunk": <original record>, "parsed": <analysis>}, one per input chunk.
All original metadata is preserved. No tokenizer, model, or repository is loaded.
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import tree_sitter_language_pack as tslp

from ingest.chunker import DEFINITION_RE, DOC_LANGUAGES, HEADING_RE, detect_language


def _source_text(chunk: dict) -> str:
    """Remove only the exact metadata header emitted by the existing chunker."""
    header = f"# {chunk.get('repo')} {chunk['path']}:{chunk['start_line']}-{chunk.get('end_line')}"
    if chunk.get("symbol"):
        header += f" {chunk['symbol']}"
    first, separator, rest = chunk["text"].partition("\n")
    return rest if separator and first == header else chunk["text"]


def _headings(text: str, start_line: int) -> list[dict]:
    headings = []
    fence = None
    for offset, line in enumerate(text.splitlines()):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not tail.strip():
                fence = None
            continue
        match = HEADING_RE.match(line) if fence is None else None
        if match:
            headings.append({"title": match[2], "level": len(match[1]), "line": start_line + offset})
    return headings


def parse_chunk(chunk: dict) -> dict:
    """Return an unchanged chunk plus syntax information found within it.

    Positions refer to the original file using the supplied start_line. For
    fragments, partial means Tree-sitter found a syntax error; it does not prove
    a chunk is complete when no error is found. No missing context is invented.
    """
    if not isinstance(chunk, dict):
        raise ValueError("each chunk must be a JSON object")
    for key in ("text", "path"):
        if not isinstance(chunk.get(key), str) or not chunk[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    start = chunk.get("start_line")
    if type(start) is not int or start < 1:
        raise ValueError("start_line must be a positive integer")
    if "end_line" in chunk and (type(chunk["end_line"]) is not int or chunk["end_line"] < start):
        raise ValueError("end_line must be an integer at least start_line")
    if "language" in chunk and (not isinstance(chunk["language"], str) or not chunk["language"]):
        raise ValueError("language must be a nonempty string")

    fallback = "text" if Path(chunk["path"]).suffix.lower() == ".txt" else None
    language = chunk.get("language") or fallback or detect_language(chunk["path"])
    source = _source_text(chunk)
    parsed = {"language": language, "status": "ok", "definitions": [], "headings": []}
    result = {"chunk": dict(chunk), "parsed": parsed}
    if language in DOC_LANGUAGES:
        if language == "markdown":
            parsed["headings"] = _headings(source, start)
        else:
            parsed["status"] = "text"
        return result

    try:
        parser = tslp.get_parser(language)
    except Exception:
        # Grammar downloads can fail offline; retain the chunk and report it.
        parsed["status"] = "unavailable"
        return result
    root = parser.parse(source.encode("utf-8")).root_node
    parsed["status"] = "partial" if root.has_error else "ok"

    stack = [(root, [])]
    while stack:
        node, scope = stack.pop()
        name_node = node.child_by_field_name("name")
        if name_node is None and node.type == "impl_item":
            name_node = node.child_by_field_name("type")
        if DEFINITION_RE.search(node.type) and name_node is not None:
            name = name_node.text.decode("utf-8")
            scope = scope + [name]
            last_row = node.end_point[0] - (node.end_point[1] == 0 and node.end_point[0] > node.start_point[0])
            parsed["definitions"].append({
                "name": ".".join(scope),
                "type": node.type,
                "start_line": start + node.start_point[0],
                "end_line": start + last_row,
            })
        stack.extend((child, scope) for child in reversed(node.named_children))
    return result


def parse_jsonl(input_path: Path, output_path: Path) -> int:
    """Stream chunks to a separate output file; preserve prior output on failure."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    temporary = None
    count = 0
    try:
        with input_path.open(encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False) as target:
                temporary = Path(target.name)
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        record = parse_chunk(json.loads(line))
                    except ValueError as error:
                        raise ValueError(f"input line {line_number}: {error}") from error
                    target.write(json.dumps(record) + "\n")
                    count += 1
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunks", type=Path, help="JSONL from the chunker")
    parser.add_argument("--out", type=Path, default=Path("parsed_chunks.jsonl"))
    args = parser.parse_args()
    try:
        count = parse_jsonl(args.chunks, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Parser failed: {error}\n")
    print(f"Parsed {count} chunks into {args.out}")


if __name__ == "__main__":
    main()
