import argparse
import json
from dataclasses import dataclass

import torch

from src.model.pre_norm_transformer_block import TransformerLM
from src.tokenizer.core import BPE
from src.training.checkpointing import load_checkpoint
from src.training.config import build_dataclass_config, load_run_config
from src.training.decoding import generate_text
from src.training.experiment import ExperimentLogger


@dataclass
class GenerationConfig:
    checkpoint_path: str
    vocab_path: str
    merges_path: str
    prompt: str
    results_dir: str
    run_name: str
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    vocab_size: int
    context_length: int
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    device: str = "cpu"
    special_token: list[str] | None = None
    theta: float = 10000.0


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--vocab-path")
    parser.add_argument("--merges-path")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--device")
    parser.add_argument("--results-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--theta", type=float)
    args = parser.parse_args()
    config_data = load_run_config(args.config)
    return build_dataclass_config(GenerationConfig, config_data, vars(args))


def main() -> None:
    args = parse_args()
    tokenizer = BPE.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_token or [],
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
