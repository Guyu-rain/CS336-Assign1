import argparse
from dataclasses import dataclass

import numpy as np

from src.tokenizer.core import BPE
from src.training.config import build_dataclass_config, load_run_config


@dataclass
class PrepareDataConfig:
    input_path: str
    output_path: str
    vocab_path: str
    merges_path: str
    special_token: list[str] | None = None
    dtype: str = "int32"


def parse_args() -> PrepareDataConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--vocab-path")
    parser.add_argument("--merges-path")
    parser.add_argument("--dtype")
    args = parser.parse_args()
    config_data = load_run_config(args.config)
    return build_dataclass_config(PrepareDataConfig, config_data, vars(args))


def main() -> None:
    args = parse_args()
    tokenizer = BPE.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_token or [],
    )

    with open(args.input_path, "r", encoding="utf-8") as handle:
        token_ids = list(tokenizer.encode_iterable(handle))

    array = np.asarray(token_ids, dtype=args.dtype)
    np.save(args.output_path, array)


if __name__ == "__main__":
    main()
