# Parse chunks

This parser takes chunks that have already been created. It does not read a
repository, split files, generate embeddings, or modify the existing chunker.

```bash
uv run python -m ingest.parser chunks.jsonl --out parsed_chunks.jsonl
```

Each input line is a JSON object with `text`, `path`, and `start_line` (a positive,
one-based original file line number). All other fields are preserved. The existing
chunker's full records work directly; their metadata header is excluded from syntax
analysis without changing the stored text or shifting original source positions.

For example, an input chunk can be:

```json
{"text":"def hello():\n    return 'hi'\n","path":"src/demo.py","start_line":20}
```

Each output line contains two fields:

- `chunk`: the complete original input record, unchanged. Embeddings can consume
  `record["chunk"]["text"]`; original IDs, citations and token counts remain here.
- `parsed`: detected language, parse status, named definitions with original file
  line ranges, and Markdown headings with levels and original line numbers.

The output is a separate analysis format. The nested `chunk` retains the existing
chunk contract; the outer wrapper is not a replacement for that contract.

Python usage:

```python
from ingest.parser import parse_chunk

result = parse_chunk({
    "text": "def hello():\n    return 'hi'\n",
    "path": "src/demo.py",
    "start_line": 20,
})
print(result["parsed"]["definitions"])
```

Code is analyzed using the project's existing Tree-sitter dependency. Grammars may
download on first use. No tokenizer or model is loaded. Definitions include names
such as `Greeter.greet` when the enclosing class is present in the same chunk.

Statuses are `ok` (no syntax errors reported), `partial` (syntax errors, possibly
because the chunk cuts through code), `unavailable` (grammar could not be loaded),
and `text` (plain text or RST, retained without structural analysis). `ok` does not
guarantee a fragment is complete. Chunks are never dropped because parsing is partial
or unavailable. Missing surrounding classes, Markdown fences, or headings cannot be
recovered from an isolated chunk. Only `#`-style Markdown headings are extracted.

Invalid input reports its JSONL line number and leaves previous output intact.
Input and output must be different paths. Filtering and secret scanning belong
upstream; this parser preserves the chunks it receives.

```bash
uv run pytest tests/test_parser.py
```
