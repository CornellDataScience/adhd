# app

Static CDS-inspired frontend for the project assistant. It includes the chat shell,
project discovery cards, and a client-side connection to the prototype `/generate`
endpoint.

## Run locally

From the repository root:

```bash
uv run uvicorn inference.prototype_server:app --reload
python3 -m http.server 3000 --directory app
```

Open http://127.0.0.1:3000. The model server should be running on port 8000 for
questions to generate answers; without it, the interface shows a connection message.
