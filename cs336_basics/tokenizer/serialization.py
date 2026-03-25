from __future__ import annotations

import base64
import json
import os


VOCAB_FORMAT = "cs336_bpe_vocab"
MERGES_FORMAT = "cs336_bpe_merges"


def bytes_to_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def b64_to_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def save_vocab_json(vocab: dict[int, bytes], vocab_filepath: str) -> None:
    data = {
        "format": VOCAB_FORMAT,
        "version": 1,
        "vocab": {str(index): bytes_to_b64(token) for index, token in vocab.items()},
    }
    os.makedirs(os.path.dirname(vocab_filepath), exist_ok=True)
    with open(vocab_filepath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_merges_json(merges: list[tuple[bytes, bytes]], merges_filepath: str) -> None:
    data = {
        "format": MERGES_FORMAT,
        "version": 1,
        "merges": [[bytes_to_b64(first), bytes_to_b64(second)] for first, second in merges],
    }
    os.makedirs(os.path.dirname(merges_filepath), exist_ok=True)
    with open(merges_filepath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_bpe_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str,
    merges_filepath: str,
) -> None:
    save_vocab_json(vocab, vocab_filepath)
    save_merges_json(merges, merges_filepath)


def load_vocab_json(vocab_filepath: str, expected_format: str = VOCAB_FORMAT) -> dict[int, bytes]:
    with open(vocab_filepath, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("format") != expected_format:
        raise ValueError(f"Unexpected vocab format: {data.get('format')}")
    return {int(index): b64_to_bytes(token_b64) for index, token_b64 in data["vocab"].items()}


def load_merges_json(
    merges_filepath: str,
    expected_format: str = MERGES_FORMAT,
) -> list[tuple[bytes, bytes]]:
    with open(merges_filepath, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("format") != expected_format:
        raise ValueError(f"Unexpected merges format: {data.get('format')}")
    return [
        (b64_to_bytes(first_b64), b64_to_bytes(second_b64))
        for first_b64, second_b64 in data["merges"]
    ]

