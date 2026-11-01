import copy
import json

import pytest

from ingest.chunker import chunk_file
from ingest.parser import parse_chunk, parse_jsonl
from tests.test_contracts import validator


def chunk(text, path="example.py", start_line=10, **metadata):
    return {"text": text, "path": path, "start_line": start_line, **metadata}


def test_python_definitions_preserve_chunk_and_file_positions():
    original = chunk("class Greeter:\n    def greet(self):\n        return 'hello'\n", id="unchanged")
    before = copy.deepcopy(original)
    result = parse_chunk(original)
    assert original == before == result["chunk"]
    assert result["parsed"]["status"] == "ok"
    assert result["parsed"]["definitions"] == [
        {"name": "Greeter", "type": "class_definition", "start_line": 10, "end_line": 12},
        {"name": "Greeter.greet", "type": "function_definition", "start_line": 11, "end_line": 12},
    ]


def test_accepts_existing_chunker_records_without_header_line_offset():
    source = b"# documentation\ndef hello():\n    return 'hello'\n"
    records = chunk_file("org/repo", "example.py", source, "a" * 40, lambda s: len(s.split()))
    for original in records:
        result = parse_chunk(original)
        validator("chunk.schema.json").validate(result["chunk"])
        assert result["chunk"] == original
        assert result["parsed"]["definitions"][0]["start_line"] == 2


def test_markdown_heading_positions_and_fences():
    original = chunk("# Project\n````python\n```\n# hidden\n````\n## Setup\n", path="README.md")
    result = parse_chunk(original)
    assert result["parsed"]["headings"] == [
        {"title": "Project", "level": 1, "line": 10},
        {"title": "Setup", "level": 2, "line": 15},
    ]
    assert result["chunk"]["text"] == original["text"]


def test_partial_code_is_preserved_and_flagged():
    original = chunk("def unfinished(\n")
    result = parse_chunk(original)
    assert result["parsed"]["status"] == "partial"
    assert result["chunk"] == original


def test_grammar_unavailable_does_not_drop_chunk(monkeypatch):
    from ingest import parser

    def unavailable(language):
        raise RuntimeError("offline")

    monkeypatch.setattr(parser.tslp, "get_parser", unavailable)
    original = chunk("some code", language="python")
    assert parse_chunk(original)["parsed"]["status"] == "unavailable"
    assert parse_chunk(original)["chunk"] == original


def test_plain_text_preserved():
    original = chunk("Project overview.", path="notes.txt")
    result = parse_chunk(original)
    assert result["parsed"]["status"] == "text"
    assert result["chunk"] == original


@pytest.mark.parametrize("bad", [None, [], {}, chunk(""), chunk("x", start_line=0), chunk("x", start_line=True), chunk("x", end_line=3), chunk("x", language=12)])
def test_rejects_invalid_chunk(bad):
    with pytest.raises(ValueError):
        parse_chunk(bad)


def test_jsonl_one_output_per_input_and_atomic_failure(tmp_path):
    source = tmp_path / "chunks.jsonl"
    target = tmp_path / "parsed.jsonl"
    chunks = [chunk("# Project\n", path="README.md"), chunk("def run():\n    pass\n")]
    source.write_text("\n".join(json.dumps(c) for c in chunks) + "\n\n")
    assert parse_jsonl(source, target) == 2
    saved = target.read_text()
    assert [json.loads(line)["chunk"] for line in saved.splitlines()] == chunks
    source.write_text(source.read_text() + "{broken\n")
    with pytest.raises(ValueError, match="input line 4"):
        parse_jsonl(source, target)
    assert target.read_text() == saved
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunks.jsonl", "parsed.jsonl"]
    with pytest.raises(ValueError, match="different files"):
        parse_jsonl(source, source)
