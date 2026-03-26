import argparse
import json

import torch

from src.model.pre_norm_transformer_block import TransformerLM
from src.tokenizer.core import BPE
from src.training.checkpointing import load_checkpoint
from src.training.decoding import generate_text
from src.training.experiment import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--merges-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", default="generation")
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--theta", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = BPE.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_token,
    )

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        theta=args.theta,
        device=torch.device(args.device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    iteration = load_checkpoint(args.checkpoint_path, model, optimizer)

    result = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        context_length=args.context_length,
    )

    logger = ExperimentLogger(
        results_dir=args.results_dir,
        run_name=args.run_name,
        config=vars(args),
    )
    logger.save_text_sample(
        "generated_text",
        prompt=args.prompt,
        generated_text=result["text"],
        metadata={
            "checkpoint_iteration": iteration,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    logger.append_experiment_note(
        "generation",
        json.dumps(
            {
                "checkpoint_iteration": iteration,
                "prompt": args.prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    print(result["text"])


if __name__ == "__main__":
    main()
