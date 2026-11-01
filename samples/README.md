# samples

`chunks.jsonl`: 100 chunks from 6 public CDS repos (adhd, FTM-Snake, slurm-aware, timelyai,
geometry-dash-rl, PredictionMarkets), valid against `contracts/chunk.schema.json`. Use it to
build and test retrieval without the cluster. Regenerate after schema or chunker changes:

```bash
uv run python -m ingest.sample .:CornellDataScience/adhd path/to/FTM-Snake:CornellDataScience/FTM-Snake ... --n 100
```
