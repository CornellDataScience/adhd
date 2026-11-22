"""Qwen3-Reranker-0.6B second stage.

Each candidate is scored as the probability of ``yes``. The prompt is the model
card's yes/no chat template. The instruction is the same code-retrieval task
the embedder uses. Documents are the stored chunk text.
"""

from collections.abc import Sequence

from retrieval.embed import TASK

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MAX_LENGTH = 8192
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_loaded: dict = {}


def format_document(question: str, document: str, task: str = TASK) -> str:
    """User block for one candidate. The model card puts a space after each colon."""
    return f"<Instruct>: {task}\n<Query>: {question}\n<Document>: {document}"


def rerank(question: str, documents: Sequence[str]) -> list[float]:
    """Score ``documents`` for ``question``. One probability per document, in order.

    Loads the model on first use. An empty list does not load it.
    """
    if not documents:
        return []
    return _score([format_document(question, document) for document in documents])


def _score(pairs: list[str]) -> list[float]:
    import torch

    tokenizer, model, prefix_tokens, suffix_tokens, token_true_id, token_false_id = _load()
    usable = MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    tokenized = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=usable,
    )
    for index, token_ids in enumerate(tokenized["input_ids"]):
        tokenized["input_ids"][index] = prefix_tokens + token_ids + suffix_tokens
    batch = tokenizer.pad(tokenized, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits[:, -1, :]
        false_vector = logits[:, token_false_id]
        true_vector = logits[:, token_true_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        yes_probability = torch.nn.functional.log_softmax(stacked, dim=1).exp()
    return yes_probability[:, 1].detach().float().cpu().tolist()


def _load():
    if "model" not in _loaded:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model.to(_device())
        model.eval()
        _loaded["tokenizer"] = tokenizer
        _loaded["model"] = model
        _loaded["prefix_tokens"] = tokenizer.encode(PREFIX, add_special_tokens=False)
        _loaded["suffix_tokens"] = tokenizer.encode(SUFFIX, add_special_tokens=False)
        _loaded["token_true_id"] = tokenizer.convert_tokens_to_ids("yes")
        _loaded["token_false_id"] = tokenizer.convert_tokens_to_ids("no")
    return (
        _loaded["tokenizer"],
        _loaded["model"],
        _loaded["prefix_tokens"],
        _loaded["suffix_tokens"],
        _loaded["token_true_id"],
        _loaded["token_false_id"],
    )


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
