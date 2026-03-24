import regex as re
from typing import Iterable, Iterator
from __init__ import DATA_PATH, PAT

def word_to_token_tuple(
    word: str,
) -> tuple[bytes, ...]:
    bytes_sequence = word.encode("utf-8")
    return tuple(bytes([b]) for b in bytes_sequence)

def pretokenize_to_str(
    text: str,
    special_tokens: list[str],
    vocab: dict[int, bytes],
    pattern: str=PAT,
) -> list[str]:
    if special_tokens:
        escaped = sorted(
            (re.escape(tok) for tok in special_tokens),
            key=len,
            reverse=True,
        )
        special_tok_pattern = "|".join(escaped)
        combined_pattern = re.compile(f"{special_tok_pattern}|{pattern}")
    else:
        combined_pattern = re.compile(pattern)

    return [m.group(0) for m in combined_pattern.finditer(text)]

class BPE:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.token_to_id = {token: idx for idx, token in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        self.special_token_to_id:dict[str, int] = {}
        self.id_to_special_token:dict[int, str] = {}

        for i, tok in enumerate(self.special_tokens):
            # 检查 vocab 的初始化是否正确
            assert vocab[i] == tok.encode("utf-8")
            self.special_token_to_id[tok] = i
            self.id_to_special_token[i] = tok

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        pass

    # "the cat ate" -> [9, 7, 1, 5, 10, 3]
    def encode(
        self, 
        text: str
    ) -> list[int]:
        ids = []
        for piece in pretokenize_to_str(text, self.special_tokens):
            if piece in self.special_tokens:
                ids.append(self.special_token_to_id[piece])
            else:
                ids.append(word_to_token_tuple(piece))
        # 执行 merge
        for piece in ids:
            if isinstance(piece, int):
                continue
        return ids

    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        pass

    def decode(
        self,
        ids: list[int]
    ) -> str:
        pass