# contracts

JSON Schemas that every part of the pipeline codes against. Change them in their own PR.

- `chunk.schema.json`: one line of `chunks.jsonl` / one row of the `chunks` table
- `retrieve.schema.json`: `/retrieve` request and response
- `generate.schema.json`: `/generate` request and response (same for Mamba-3 and the Transformer)
