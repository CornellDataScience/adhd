"""Build the numbered-source prompt from a retrieve-contract response."""

from __future__ import annotations

DEFAULT_INSTRUCTION = (
    "Answer the question using only the sources below. "
    "Cite sources as [1], [2], and so on. "
    "If the sources do not contain the answer, say you do not know."
)


def build_prompt(
    retrieve_response: dict,
    *,
    question: str | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    """Turn a ``contracts/retrieve.schema.json`` response into a generation prompt.

    Sources are listed in rank order. ``rank`` 1 becomes citation ``[1]``.
    """
    q = question if question is not None else retrieve_response.get("question", "")
    if not q:
        raise ValueError("question is required")
    results = retrieve_response.get("results") or []
    lines = [instruction, "", f"Question: {q}", ""]
    if not results:
        lines.append("Sources: none")
    else:
        lines.append("Sources:")
        for hit in results:
            rank = hit["rank"]
            chunk = hit["chunk"]
            header = f"[{rank}] {chunk.get('repo', '')} {chunk.get('path', '')}"
            if chunk.get("symbol"):
                header += f" ({chunk['symbol']})"
            lines.append(header)
            lines.append(chunk.get("text", "").rstrip())
            lines.append("")
    lines.append("Answer:")
    return "\n".join(lines).rstrip() + "\n"
