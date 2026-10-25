import textwrap

from ingest.chunker import chunk_file
from tests.test_contracts import validator

SHA = "0" * 40


def words(text):
    return len(text.split())


def chunk(path, source, min_tokens=20, max_tokens=60):
    source = textwrap.dedent(source).lstrip("\n")
    return chunk_file("org/repo", path, source.encode(), SHA, words, min_tokens, max_tokens), source


def covered_lines(records):
    lines = []
    for r in records:
        lines.extend(range(r["start_line"], r["end_line"] + 1))
    return lines


PYTHON = '''
import os

# Adds things.
@decorator
def add(a, b):
    return a + b


class Greeter:
    """Says hello."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        word = "hello"
        other = "there"
        return f"{word} {other} {self.name}, nice to meet you today in this long sentence"

    def wave(self):
        return "wave wave wave wave wave wave wave wave wave wave wave wave wave"


def long_function():
    a = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15

    b = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15

    c = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15
    return a + b + c
'''


def test_python_chunks_follow_definitions():
    records, source = chunk("pkg/mod.py", PYTHON)
    assert all(r["token_count"] <= 60 for r in records)
    assert all(r["language"] == "python" and r["kind"] == "code" for r in records)

    # Every non-blank line is in exactly one chunk.
    lines = covered_lines(records)
    assert len(lines) == len(set(lines))
    nonblank = {i + 1 for i, line in enumerate(source.split("\n")) if line.strip()}
    assert nonblank <= set(lines)

    symbols = " | ".join(r["symbol"] or "" for r in records)
    assert "add" in symbols
    assert "Greeter" in symbols
    long_parts = [r for r in records if r["symbol"] and "long_function" in r["symbol"]]
    assert len(long_parts) >= 2  # split, and every piece keeps the name


def test_continuation_headers_repeat_the_signature():
    records, _ = chunk("pkg/mod.py", PYTHON)
    first, *rest = [r for r in records if r["symbol"] and "long_function" in r["symbol"]]
    assert " | " not in first["text"].split("\n")[0]  # the signature is already in the code
    assert rest and all(r["text"].split("\n")[0].endswith(" | def long_function()") for r in rest)


def test_multiline_signature_is_normalized():
    source = "def f(\n    a,\n    b,\n):\n" + "".join(f"    x{i} = {i} + {i} + {i} + {i}\n\n" for i in range(12))
    records, _ = chunk("pkg/f.py", source)
    assert records[1]["text"].split("\n")[0].endswith(" | def f(a, b)")


def test_oversized_class_splits_into_qualified_methods():
    records, _ = chunk("pkg/mod.py", PYTHON, min_tokens=5, max_tokens=25)
    symbols = " | ".join(r["symbol"] or "" for r in records)
    assert "Greeter.greet" in symbols and "Greeter.wave" in symbols


def test_leading_comment_and_decorator_stay_with_definition():
    records, _ = chunk("pkg/mod.py", PYTHON)
    add = next(r for r in records if r["symbol"] and "add" in r["symbol"].split(", "))
    assert "# Adds things.\n@decorator\ndef add" in add["text"]


def test_small_units_are_merged():
    records, _ = chunk("pkg/mod.py", PYTHON, min_tokens=1000, max_tokens=1000)
    assert len(records) == 1


def test_header_and_metadata():
    records, _ = chunk("pkg/mod.py", PYTHON)
    r = records[0]
    assert r["text"].startswith(f"# org/repo pkg/mod.py:{r['start_line']}-{r['end_line']}")
    assert r["source_url"] == f"https://github.com/org/repo/blob/{SHA}/pkg/mod.py#L{r['start_line']}-L{r['end_line']}"
    assert len({r["id"] for r in records}) == len(records)
    assert len({r["file_hash"] for r in records}) == 1
    v = validator("chunk.schema.json")
    for r in records:
        v.validate(r)


MARKDOWN = """
# Title

Intro text.

## Setup

Run this:

```bash
# not a heading
uv sync
```

### Cluster

On the cluster.
"""


def test_markdown_splits_on_headings_outside_code_fences():
    records, _ = chunk("README.md", MARKDOWN, min_tokens=1, max_tokens=60)
    assert [r["symbol"] for r in records] == ["Title", "Title > Setup", "Title > Setup > Cluster"]
    assert all(r["kind"] == "readme" and r["language"] == "markdown" for r in records)
    assert "# not a heading" in records[1]["text"]


def test_rust_impl_methods_are_qualified():
    rust = """
    struct Game { score: u32 }

    impl Game {
        fn new() -> Self {
            Game { score: 0 }
        }

        fn tick(&mut self) {
            self.score += 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12;
        }
    }
    """
    records, _ = chunk("src/game.rs", rust, min_tokens=1, max_tokens=30)
    symbols = " | ".join(r["symbol"] or "" for r in records)
    assert "Game.new" in symbols and "Game.tick" in symbols


def test_single_huge_line_is_kept_whole():
    records, _ = chunk("data.txt", "word " * 500)
    assert len(records) == 1 and records[0]["token_count"] > 60


def test_blank_file_has_no_chunks():
    assert chunk("empty.py", "\n\n")[0] == []
