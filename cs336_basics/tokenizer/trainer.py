from __future__ import annotations

import collections
import os
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re
from tqdm import tqdm

from .constants import DEFAULT_PRETOKENIZATION_PATTERN, DEFAULT_TRAINING_DATA_PATH
from .pretokenization import text_to_byte_tuple, split_on_special_tokens
from .serialization import (
    b64_to_bytes,
    bytes_to_b64,
    load_merges_json,
    load_vocab_json,
    save_bpe_tokenizer,
    save_merges_json,
    save_vocab_json,
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(
    args: tuple[str, int, int, list[str], str]
) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens, pattern = args
    with open(input_path, "rb") as handle:
        handle.seek(start)
        chunk_text = handle.read(end - start).decode("utf-8", errors="ignore")

    local_corpus: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    special_token_set = set(special_tokens)
    for part in split_on_special_tokens(chunk_text, special_tokens):
        if part in special_token_set:
            continue
        for match in re.finditer(pattern, part):
            local_corpus[text_to_byte_tuple(match.group())] += 1
    return dict(local_corpus)


def _merge_corpus_dicts(
    dicts: list[dict[tuple[bytes, ...], int]]
) -> dict[tuple[bytes, ...], int]:
    merged: dict[tuple[bytes, ...], int] = collections.defaultdict(int)
    for current in dicts:
        for token_sequence, frequency in current.items():
            merged[token_sequence] += frequency
    return merged


def parallel_pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_workers: int,
    pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as handle:
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
        boundaries = find_chunk_boundaries(handle, num_workers, split_token)

    chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))
    args = [(input_path, start, end, special_tokens, pattern) for start, end in chunk_boundaries]
    with Pool(processes=num_workers) as pool:
        partial_corpora = pool.map(_pretokenize_chunk, args)
    return _merge_corpus_dicts(partial_corpora)


def _build_pair_counts(
    corpus: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    for token_sequence, frequency in corpus.items():
        for first, second in zip(token_sequence, token_sequence[1:]):
            pair_counts[(first, second)] += frequency
    return pair_counts


def _apply_merge(
    corpus: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    best_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    first, second = best_pair
    merged_token = first + second
    new_corpus: dict[tuple[bytes, ...], int] = {}

    for token_sequence, frequency in corpus.items():
        if first not in token_sequence or not any(
            token_sequence[index] == first and token_sequence[index + 1] == second
            for index in range(len(token_sequence) - 1)
        ):
            new_corpus[token_sequence] = frequency
            continue

        new_sequence: list[bytes] = []
        index = 0
        while index < len(token_sequence):
            if (
                index < len(token_sequence) - 1
                and token_sequence[index] == first
                and token_sequence[index + 1] == second
            ):
                if new_sequence:
                    pair_counts[(new_sequence[-1], first)] -= frequency
                    pair_counts[(new_sequence[-1], merged_token)] += frequency
                if index + 2 < len(token_sequence):
                    pair_counts[(second, token_sequence[index + 2])] -= frequency
                    pair_counts[(merged_token, token_sequence[index + 2])] += frequency

                pair_counts[best_pair] -= frequency
                new_sequence.append(merged_token)
                index += 2
            else:
                new_sequence.append(token_sequence[index])
                index += 1

        new_key = tuple(new_sequence)
        new_corpus[new_key] = new_corpus.get(new_key, 0) + frequency

    return new_corpus


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
    pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if num_workers is None:
        num_workers = cpu_count()

    vocab: dict[int, bytes] = {}
    vocab_index = 0
    for token in special_tokens:
        vocab[vocab_index] = token.encode("utf-8")
        vocab_index += 1
    for value in range(256):
        vocab[vocab_index] = bytes([value])
        vocab_index += 1

    min_vocab_size = len(special_tokens) + 256
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size}")

    num_merges = vocab_size - min_vocab_size
    if num_merges == 0:
        return vocab, []

    corpus = parallel_pretokenize(input_path, special_tokens, num_workers, pattern=pattern)
    pair_counts = _build_pair_counts(corpus)
    merges: list[tuple[bytes, bytes]] = []

    for _ in tqdm(range(num_merges), desc="Training BPE", unit="merge"):
        if not pair_counts:
            break

        best_pair = max(
            ((pair, count) for pair, count in pair_counts.items() if count > 0),
            key=lambda item: (item[1], item[0]),
            default=None,
        )
        if best_pair is None:
            break

        best_pair_key = best_pair[0]
        merges.append(best_pair_key)
        vocab[vocab_index] = best_pair_key[0] + best_pair_key[1]
        vocab_index += 1
        corpus = _apply_merge(corpus, pair_counts, best_pair_key)

    return vocab, merges


__all__ = [
    "DEFAULT_TRAINING_DATA_PATH",
    "b64_to_bytes",
    "bytes_to_b64",
    "find_chunk_boundaries",
    "load_merges_json",
    "load_vocab_json",
    "parallel_pretokenize",
    "save_bpe_tokenizer",
    "save_merges_json",
    "save_vocab_json",
    "train_bpe",
]
