import argparse
from dataclasses import dataclass

from .constants import DEFAULT_PRETOKENIZATION_PATTERN
from .trainer import (
    DEFAULT_TRAINING_DATA_PATH as DATA_PATH,
    b64_to_bytes as _b64_to_bytes,
    bytes_to_b64 as _bytes_to_b64,
    find_chunk_boundaries as _find_chunk_boundaries,
    load_merges_json,
    load_vocab_json,
    parallel_pretokenize,
    save_bpe_tokenizer,
    save_merges_json,
    save_vocab_json,
    train_bpe,
)
from .pretokenization import text_to_byte_tuple as word_to_token_tuple
from src.training.config import build_dataclass_config, load_run_config


@dataclass
class TrainBPEConfig:
    input_path: str
    vocab_size: int
    special_tokens: list[str]
    vocab_output_path: str
    merges_output_path: str
    num_workers: int | None = None
    pattern: str | None = None


def parse_args() -> TrainBPEConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-path")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--vocab-output-path")
    parser.add_argument("--merges-output-path")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pattern")
    args = parser.parse_args()
    config_data = load_run_config(args.config)
    return build_dataclass_config(TrainBPEConfig, config_data, vars(args))


def main() -> None:
    config = parse_args()
    vocab, merges = train_bpe(
        input_path=config.input_path,
        vocab_size=config.vocab_size,
        special_tokens=config.special_tokens,
        num_workers=config.num_workers,
        pattern=config.pattern or DEFAULT_PRETOKENIZATION_PATTERN,
    )
    save_bpe_tokenizer(
        vocab=vocab,
        merges=merges,
        vocab_filepath=config.vocab_output_path,
        merges_filepath=config.merges_output_path,
    )

__all__ = [
    "DATA_PATH",
    "_b64_to_bytes",
    "_bytes_to_b64",
    "_find_chunk_boundaries",
    "load_merges_json",
    "load_vocab_json",
    "parallel_pretokenize",
    "save_bpe_tokenizer",
    "save_merges_json",
    "save_vocab_json",
    "train_bpe",
    "word_to_token_tuple",
]


if __name__ == "__main__":
    main()
