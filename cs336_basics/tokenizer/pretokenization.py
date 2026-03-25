from __future__ import annotations

from collections.abc import Iterable

import regex as re

from .constants import DEFAULT_PRETOKENIZATION_PATTERN


def text_to_byte_list(text: str) -> list[bytes]:
    return [bytes([value]) for value in text.encode("utf-8")]


def text_to_byte_tuple(text: str) -> tuple[bytes, ...]:
    return tuple(text_to_byte_list(text))


def split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
    if not special_tokens:
        return [text]

    escaped_specials = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)
    special_pattern = re.compile("(" + "|".join(escaped_specials) + ")")
    return [part for part in special_pattern.split(text) if part]


def pretokenize_text(
    text: str,
    special_tokens: list[str] | None = None,
    pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
) -> list[str]:
    if not special_tokens:
        return [match.group(0) for match in re.finditer(pattern, text)]

    special_token_set = set(special_tokens)
    pieces: list[str] = []
    for part in split_on_special_tokens(text, special_tokens):
        if part in special_token_set:
            pieces.append(part)
        else:
            pieces.extend(match.group(0) for match in re.finditer(pattern, part))
    return pieces


def iter_pretokenized_text(
    texts: Iterable[str],
    special_tokens: list[str] | None = None,
    pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
):
    for text in texts:
        yield from pretokenize_text(text, special_tokens=special_tokens, pattern=pattern)

