from __future__ import annotations

from collections.abc import Iterable, Iterator

from .constants import DEFAULT_PRETOKENIZATION_PATTERN
from .pretokenization import pretokenize_text, text_to_byte_list
from .serialization import MERGES_FORMAT, VOCAB_FORMAT, load_merges_json, load_vocab_json


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.pattern = pattern

        self.token_to_id = {token: index for index, token in vocab.items()}
        self.merge_ranks = {pair: index for index, pair in enumerate(merges)}

        self.special_token_to_id: dict[str, int] = {}
        self.id_to_special_token: dict[int, str] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                raise ValueError(f"Special token {token!r} is missing from the vocabulary")
            token_id = self.token_to_id[token_bytes]
            self.special_token_to_id[token] = token_id
            self.id_to_special_token[token_id] = token

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
        vocab_format: str = VOCAB_FORMAT,
        merges_format: str = MERGES_FORMAT,
        pattern: str = DEFAULT_PRETOKENIZATION_PATTERN,
    ) -> "BPE":
        vocab = load_vocab_json(vocab_filepath, expected_format=vocab_format)
        merges = load_merges_json(merges_filepath, expected_format=merges_format)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, pattern=pattern)

    def _merge_piece(self, piece: str) -> list[bytes]:
        piece_tokens = list(text_to_byte_list(piece))
        for first, second in self.merges:
            new_piece: list[bytes] = []
            index = 0
            while index < len(piece_tokens):
                if (
                    index < len(piece_tokens) - 1
                    and piece_tokens[index] == first
                    and piece_tokens[index + 1] == second
                ):
                    new_piece.append(first + second)
                    index += 2
                else:
                    new_piece.append(piece_tokens[index])
                    index += 1
            piece_tokens = new_piece
        return piece_tokens

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []
        for piece in pretokenize_text(text, self.special_tokens, self.pattern):
            if piece in self.special_token_to_id:
                encoded_ids.append(self.special_token_to_id[piece])
            else:
                encoded_ids.extend(self.token_to_id[token] for token in self._merge_piece(piece))
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")


BPETokenizer = BPE

