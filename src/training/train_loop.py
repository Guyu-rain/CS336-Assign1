import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import time

import torch

from src.model.pre_norm_transformer_block import TransformerLM
from src.training.checkpointing import save_checkpoint
from src.training.config import build_dataclass_config, load_run_config
from src.training.data import get_batch, load_token_dataset
from src.training.experiment import ExperimentLogger
from src.training.loss import cross_entropy
from src.training.optimizer import AdamW, gradient_clipping, learning_rate_schedule


@dataclass
class TrainingConfig:
    train_data_path: str
    val_data_path: str | None
    checkpoint_path: str | None
    results_dir: str
    run_name: str
    batch_size: int
    context_length: int
    num_iterations: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    vocab_size: int
    device: str = "cpu"
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 0
    cosine_cycle_iters: int = 0
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    log_every: int = 10
    eval_every: int = 100
    eval_iters: int = 10
    checkpoint_every: int = 100
    compile_model: bool = False


def evaluate_loss(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model_was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            losses.append(cross_entropy(logits, y).item())
    if model_was_training:
        model.train()
    return sum(losses) / len(losses)


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    *,
    val_dataset=None,
    num_iterations: int,
    batch_size: int,
    context_length: int,
    device: str,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    grad_clip: float | None = None,
    log_every: int = 10,
    eval_every: int = 100,
    eval_iters: int = 10,
    checkpoint_every: int | None = None,
    checkpoint_path: str | None = None,
    logger: ExperimentLogger | None = None,
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []
    start_time = time.time()

    for iteration in range(num_iterations):
        lr = learning_rate_schedule(
            iteration,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)

        x, y = get_batch(train_dataset, batch_size, context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        if grad_clip is not None:
            gradient_clipping(list(model.parameters()), grad_clip)

        optimizer.step()

        metrics = {
            "iteration": float(iteration),
            "train_loss": float(loss.item()),
            "learning_rate": float(lr),
            "wallclock_time": float(time.time() - start_time),
        }

        should_eval = val_dataset is not None and (
            iteration % eval_every == 0 or iteration == num_iterations - 1
        )
        if should_eval:
            metrics["val_loss"] = evaluate_loss(
                model,
                val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_iters=eval_iters,
            )

        if iteration % log_every == 0 or iteration == num_iterations - 1:
            print(metrics)

        logs.append(metrics)
        if logger is not None:
            logger.log_metrics(metrics)

        should_save = (
            checkpoint_path is not None
            and checkpoint_every is not None
            and (iteration + 1) % checkpoint_every == 0
        )
        if should_save:
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)

    return logs


def build_model(config: TrainingConfig) -> TransformerLM:
    model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        device=torch.device(config.device),
    )
    if config.compile_model:
        if config.device == "mps":
            return torch.compile(model, backend="aot_eager")
        return torch.compile(model)
    return model


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    return AdamW(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data-path")
    parser.add_argument("--val-data-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--results-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--num-iterations", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--compile-model", action="store_true", default=None)
    args = parser.parse_args()
    config_data = load_run_config(args.config)
    return build_dataclass_config(TrainingConfig, config_data, vars(args))


def main() -> None:
    config = parse_args()
    train_dataset = load_token_dataset(config.train_data_path, mmap=True)
    val_dataset = None
    if config.val_data_path is not None:
        val_dataset = load_token_dataset(config.val_data_path, mmap=True)

    logger = ExperimentLogger(
        results_dir=config.results_dir,
        run_name=config.run_name,
        config=config,
    )
    model = build_model(config)
    optimizer = build_optimizer(model.parameters(), config)
    train(
        model,
        optimizer,
        train_dataset,
        val_dataset=val_dataset,
        num_iterations=config.num_iterations,
        batch_size=config.batch_size,
        context_length=config.context_length,
        device=config.device,
        max_learning_rate=config.lr,
        min_learning_rate=config.min_lr,
        warmup_iters=config.warmup_iters,
        cosine_cycle_iters=config.cosine_cycle_iters,
        grad_clip=config.grad_clip,
        log_every=config.log_every,
        eval_every=config.eval_every,
        eval_iters=config.eval_iters,
        checkpoint_every=config.checkpoint_every,
        checkpoint_path=config.checkpoint_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
