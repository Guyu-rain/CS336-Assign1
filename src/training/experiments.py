from dataclasses import replace

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
