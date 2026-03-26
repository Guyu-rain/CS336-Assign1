import argparse
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from src.model.pre_norm_transformer_block import TransformerLM
from src.training.checkpointing import save_checkpoint
from src.training.data import get_batch, load_token_dataset
from src.training.loss import cross_entropy
from src.training.optimizer import AdamW, gradient_clipping, learning_rate_schedule


@dataclass
class TrainingConfig:
    train_data_path: str
    val_data_path: str | None
    checkpoint_path: str | None
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
) -> list[dict[str, float]]:
    model.to(device)
    model.train()
    logs: list[dict[str, float]] = []

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

        should_save = (
            checkpoint_path is not None
            and checkpoint_every is not None
            and (iteration + 1) % checkpoint_every == 0
        )
        if should_save:
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)

    return logs


def build_model(config: TrainingConfig) -> TransformerLM:
    return TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        device=torch.device(config.device),
    )


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
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--val-data-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--cosine-cycle-iters", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def main() -> None:
    config = parse_args()
    train_dataset = load_token_dataset(config.train_data_path, mmap=True)
    val_dataset = None
    if config.val_data_path is not None:
        val_dataset = load_token_dataset(config.val_data_path, mmap=True)

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
    )


if __name__ == "__main__":
    main()
