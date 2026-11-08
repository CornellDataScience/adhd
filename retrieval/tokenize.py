"""Code tokenizer for retrieval v0.

Same rules for chunk text and questions:

- Lowercase.
- Split on non-letters. Digits and punctuation are separators, not tokens.
- Underscore stays inside an identifier so snake_case is one raw token.
- Split camelCase, PascalCase, and snake_case into parts.
- Keep that raw identifier as its own token, so ``load_model`` still matches exactly.
- No English stemmer.

A plain word is emitted once. ``load_model`` becomes ``load_model``, ``load``, ``model``.
"""

import re

# Boundaries only. The pieces on either side are the tokens.
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` with the v0 code tokenizer."""
    tokens: list[str] = []
    length = len(text)
    index = 0
    while index < length:
        if not _is_ident_char(text[index]):
            index += 1
            continue
        end = index + 1
        while end < length and _is_ident_char(text[end]):
            end += 1
        raw = text[index:end]
        index = end
        if not any(char.isalpha() for char in raw):
            continue
        parts = _split_identifier(raw)
        raw_token = raw.lower()
        tokens.append(raw_token)
        if parts != [raw_token]:
            tokens.extend(parts)
    return tokens


def _is_ident_char(char: str) -> bool:
    return char.isalpha() or char == "_"


def _split_identifier(raw: str) -> list[str]:
    parts: list[str] = []
    for segment in raw.split("_"):
        if not segment:
            continue
        parts.extend(part.lower() for part in _CAMEL.split(segment) if part)
    return parts
