import argparse

import numpy as np

from src.tokenizer.core import BPE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--merges-path", required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--dtype", default="int32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = BPE.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_token,
    )

    with open(args.input_path, "r", encoding="utf-8") as handle:
        token_ids = list(tokenizer.encode_iterable(handle))

    array = np.asarray(token_ids, dtype=args.dtype)
    np.save(args.output_path, array)


if __name__ == "__main__":
    main()
