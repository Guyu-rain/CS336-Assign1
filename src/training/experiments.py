import argparse
import json
from dataclasses import replace

from src.training.config import load_run_config
from src.training.experiment import ExperimentLogger
from src.training.train_loop import (
    TrainingConfig,
    build_model,
    build_optimizer,
    load_token_dataset,
    train,
)


def run_learning_rate_sweep(
    base_config: TrainingConfig,
    learning_rates: list[float],
) -> list[dict[str, float | str]]:
    train_dataset = load_token_dataset(base_config.train_data_path, mmap=True)
    val_dataset = None
    if base_config.val_data_path is not None:
        val_dataset = load_token_dataset(base_config.val_data_path, mmap=True)

    results = []
    for learning_rate in learning_rates:
        config = replace(
            base_config,
            lr=learning_rate,
            run_name=f"{base_config.run_name}_lr_{learning_rate:g}",
        )
        logger = ExperimentLogger(
            results_dir=config.results_dir,
            run_name=config.run_name,
            config=config,
        )
        model = build_model(config)
        optimizer = build_optimizer(model.parameters(), config)
        metrics = train(
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
        last_metrics = metrics[-1]
        results.append(
            {
                "run_name": config.run_name,
                "learning_rate": learning_rate,
                "final_train_loss": float(last_metrics["train_loss"]),
                "final_val_loss": float(last_metrics.get("val_loss", -1.0)),
            }
        )
        logger.append_experiment_note(
            "learning_rate_sweep",
            f"learning_rate={learning_rate}, final_metrics={last_metrics}",
        )
    return results


def run_batch_size_sweep(
    base_config: TrainingConfig,
    batch_sizes: list[int],
) -> list[dict[str, float | str]]:
    train_dataset = load_token_dataset(base_config.train_data_path, mmap=True)
    val_dataset = None
    if base_config.val_data_path is not None:
        val_dataset = load_token_dataset(base_config.val_data_path, mmap=True)

    results = []
    for batch_size in batch_sizes:
        config = replace(
            base_config,
            batch_size=batch_size,
            run_name=f"{base_config.run_name}_bs_{batch_size}",
        )
        logger = ExperimentLogger(
            results_dir=config.results_dir,
            run_name=config.run_name,
            config=config,
        )
        model = build_model(config)
        optimizer = build_optimizer(model.parameters(), config)
        metrics = train(
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
        last_metrics = metrics[-1]
        results.append(
            {
                "run_name": config.run_name,
                "batch_size": batch_size,
                "final_train_loss": float(last_metrics["train_loss"]),
                "final_val_loss": float(last_metrics.get("val_loss", -1.0)),
            }
        )
        logger.append_experiment_note(
            "batch_size_sweep",
            f"batch_size={batch_size}, final_metrics={last_metrics}",
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    experiment_type = config["experiment_type"]
    base_config = TrainingConfig(**config["base_config"])

    if experiment_type == "learning_rate_sweep":
        results = run_learning_rate_sweep(base_config, config["learning_rates"])
    elif experiment_type == "batch_size_sweep":
        results = run_batch_size_sweep(base_config, config["batch_sizes"])
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    logger = ExperimentLogger(
        results_dir=base_config.results_dir,
        run_name=f"{base_config.run_name}_{experiment_type}",
        config=config,
    )
    logger.append_experiment_note(
        experiment_type,
        json.dumps(results, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    main()
