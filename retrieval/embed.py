"""Qwen3-Embedding-0.6B query and document vectors.

Documents are the stored chunk text. Queries use the instruction wrapper from
the model card (``Query:`` has no space before the question). Both sides are
L2-normalized 1024-d vectors, so cosine similarity is a dot product.
"""

from collections.abc import Sequence

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024
MAX_LENGTH = 8192
TASK = (
    "Given a natural language question about a code repository, "
    "retrieve the code or documentation chunk that answers it"
)

_loaded: dict = {}


def format_query(question: str) -> str:
    """Wrap ``question`` the way this embedder was trained to see queries."""
    return f"Instruct: {TASK}\nQuery:{question}"


def embed_query(question: str) -> list[float]:
    """Embed one retrieval query. Loads the model on first use."""
    return _embed([format_query(question)])[0]


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed chunk texts with no instruction prefix. Loads the model on first use."""
    if not texts:
        return []
    return _embed(list(texts))


def _embed(texts: list[str]) -> list[list[float]]:
    import torch

    tokenizer, model = _load()
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
        pooled = _last_token_pool(hidden, batch["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.detach().float().cpu().tolist()


def _load():
    if "model" not in _loaded:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        model = AutoModel.from_pretrained(MODEL_ID)
        model.to(_device())
        model.eval()
        _loaded["tokenizer"] = tokenizer
        _loaded["model"] = model
    return _loaded["tokenizer"], _loaded["model"]


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _last_token_pool(hidden, attention_mask):
    """Pool the last non-padding token. Matches the Qwen3-Embedding model card."""
    import torch

    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, lengths]
