"""Utilities for saving checkpoints, histories, CSV summaries, and plots."""

import csv
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

import torch

from .tasks import get_prefixed_metric_key, get_primary_eval_metric_key, get_task_config
from .training_state import get_trainable_state


def save_json(path, payload):
    """Write JSON with UTF-8 encoding and pretty indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv_rows(path, rows, preferred_fields=None):
    """Write a list of dictionaries to CSV while keeping important columns first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    for field in preferred_fields or []:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_metric_display_name(task_name):
    """Return a human-readable name for the main evaluation metric."""
    metric_name = get_task_config(task_name)["metric"]
    display_names = {
        "accuracy": "Accuracy",
        "matthews": "Matthews Correlation",
        "pearson": "Pearson Correlation",
    }
    return display_names.get(metric_name, metric_name)


def save_trainable_checkpoint(
    model,
    output_dir,
    args,
    target_families,
    experiment_name,
    resolved_dataset_path,
    parameter_stats,
):
    """
    Save only the trainable parameters plus enough metadata to interpret the run.

    In this ASTRA SFT variant, the trainable parameters can include:
    - additive Tucker core deltas
    - multiplicative mode-transform coefficients
    - the classifier head

    Parameter-count summaries are saved twice on purpose:
    - `training_config.json` keeps them beside the full run configuration
    - `parameter_counts.json` is a small standalone file for quick inspection
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainable_state = get_trainable_state(model)
    torch.save(trainable_state, output_path / "trainable_state.pt")

    metadata = {
        "experiment_name": experiment_name,
        "model_path": args.model_path,
        "dataset_path": resolved_dataset_path,
        "glue_task": args.glue_task,
        "target_families": target_families,
        "target_modules": args.target_modules,
        "attn_ranks": list(args.attn_ranks),
        "ffn_ranks": list(args.ffn_ranks),
        "attn_alpha": args.attn_alpha,
        "ffn_alpha": args.ffn_alpha,
        "tuning_mode": args.tuning_mode,
        "multiplicative_num_bases": args.multiplicative_num_bases,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "decomposition_method": "tucker_hooi_svd_init",
        "decomposition_cache_dir": getattr(args, "_resolved_decomposition_cache_dir", None),
        "parameter_stats": parameter_stats,
    }
    save_json(output_path / "training_config.json", metadata)
    save_json(output_path / "parameter_counts.json", parameter_stats)


def extract_history_records(log_history, primary_eval_metric_key):
    """Split Trainer log history into train and eval tables for later plotting."""
    train_history = []
    eval_history = []

    for record in log_history:
        if "loss" in record and "eval_loss" not in record:
            train_row = {
                "step": float(record.get("step", len(train_history) + 1)),
                "loss": float(record["loss"]),
            }
            if record.get("epoch") is not None:
                train_row["epoch"] = float(record["epoch"])
            train_history.append(train_row)

        if "eval_loss" in record or primary_eval_metric_key in record:
            if record.get("epoch") is None:
                continue
            eval_row = {"epoch": float(record["epoch"])}
            if "eval_loss" in record:
                eval_row["eval_loss"] = float(record["eval_loss"])
            if primary_eval_metric_key in record:
                eval_row[primary_eval_metric_key] = float(record[primary_eval_metric_key])
            eval_history.append(eval_row)

    return train_history, eval_history


def prepend_initial_history(train_history, eval_history, initial_train_metrics, initial_eval_metrics, task_name):
    """Add epoch-0 metrics so the curves show where training started."""
    train_history_with_initial = list(train_history)
    eval_history_with_initial = list(eval_history)

    initial_train_loss = initial_train_metrics.get("train_init_loss")
    if initial_train_loss is not None:
        train_history_with_initial.insert(
            0,
            {
                "step": 0.0,
                "epoch": 0.0,
                "loss": float(initial_train_loss),
            },
        )

    initial_eval_metric_key = get_prefixed_metric_key(task_name, "eval_init")
    initial_eval_row = {"epoch": 0.0}
    if "eval_init_loss" in initial_eval_metrics:
        initial_eval_row["eval_loss"] = float(initial_eval_metrics["eval_init_loss"])
    if initial_eval_metric_key in initial_eval_metrics:
        initial_eval_row[get_primary_eval_metric_key(task_name)] = float(
            initial_eval_metrics[initial_eval_metric_key]
        )
    if len(initial_eval_row) > 1:
        eval_history_with_initial.insert(0, initial_eval_row)

    return train_history_with_initial, eval_history_with_initial


def save_training_plots(train_history, eval_history, output_dir, task_name, primary_eval_metric_key, experiment_name):
    """Save train-loss and validation-metric curves when matplotlib is available."""
    if plt is None:
        print("Warning: matplotlib is not installed, skipping plot generation.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in train_history if "loss" in row]
    if train_rows:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [row["step"] for row in train_rows],
            [row["loss"] for row in train_rows],
            marker="o",
            linewidth=1.8,
        )
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve ({experiment_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path / "train_loss_curve.png", dpi=200)
        plt.close()

    eval_rows = [row for row in eval_history if primary_eval_metric_key in row]
    if eval_rows:
        metric_display_name = get_metric_display_name(task_name)
        plt.figure(figsize=(8, 5))
        plt.plot(
            [row["epoch"] for row in eval_rows],
            [row[primary_eval_metric_key] for row in eval_rows],
            marker="o",
            linewidth=1.8,
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric_display_name)
        plt.title(f"Validation {metric_display_name} Curve ({experiment_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path / "eval_metric_curve.png", dpi=200)
        plt.close()


def save_history_artifacts(
    output_dir,
    train_history,
    eval_history,
    initial_train_metrics,
    initial_eval_metrics,
    final_metrics,
    task_name,
    primary_eval_metric_key,
    experiment_name,
):
    """Save JSON, CSV, and plots for one experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_json(
        output_path / "history.json",
        {
            "train_history": train_history,
            "eval_history": eval_history,
            "initial_train_metrics": initial_train_metrics,
            "initial_eval_metrics": initial_eval_metrics,
            "final_metrics": final_metrics,
        },
    )

    write_csv_rows(
        output_path / "train_loss_history.csv",
        train_history,
        preferred_fields=["step", "epoch", "loss"],
    )
    write_csv_rows(
        output_path / "eval_history.csv",
        eval_history,
        preferred_fields=["epoch", "eval_loss", primary_eval_metric_key],
    )
    save_training_plots(
        train_history=train_history,
        eval_history=eval_history,
        output_dir=output_path,
        task_name=task_name,
        primary_eval_metric_key=primary_eval_metric_key,
        experiment_name=experiment_name,
    )


def save_sweep_comparison_plot(experiment_results, output_dir, task_name, primary_eval_metric_key):
    """Plot the main validation metric of every experiment in a sweep."""
    if plt is None:
        print("Warning: matplotlib is not installed, skipping sweep comparison plot.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_slug = get_task_config(task_name)["metric"]
    metric_display_name = get_metric_display_name(task_name)

    plt.figure(figsize=(10, 6))
    line_count = 0
    for result in experiment_results:
        rows = [row for row in result["eval_history"] if primary_eval_metric_key in row]
        if not rows:
            continue
        plt.plot(
            [row["epoch"] for row in rows],
            [row[primary_eval_metric_key] for row in rows],
            marker="o",
            linewidth=1.8,
            label=result["experiment_name"],
        )
        line_count += 1

    if line_count == 0:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(metric_display_name)
    plt.title(f"{task_name.upper()} {metric_display_name} Comparison Across Experiments")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"sweep_{metric_slug}_comparison.png", dpi=220)
    plt.close()


def create_sweep_summary_rows(experiment_results, task_name):
    """Flatten sweep results into one summary table."""
    primary_eval_metric_key = get_primary_eval_metric_key(task_name)
    initial_eval_metric_key = get_prefixed_metric_key(task_name, "eval_init")

    rows = []
    for result in experiment_results:
        config = result["config"]
        initial_train_metrics = result["initial_train_metrics"]
        initial_eval_metrics = result["initial_eval_metrics"]
        final_metrics = result["final_metrics"]

        rows.append(
            {
                "experiment_name": result["experiment_name"],
                "attn_ranks": "-".join(str(x) for x in config["attn_ranks"]),
                "ffn_ranks": "-".join(str(x) for x in config["ffn_ranks"]),
                "attn_alpha": config["attn_alpha"],
                "ffn_alpha": config["ffn_alpha"],
                "tuning_mode": config.get("tuning_mode"),
                "multiplicative_num_bases": config.get("multiplicative_num_bases"),
                "num_train_epochs": config["num_train_epochs"],
                "learning_rate": config["learning_rate"],
                "train_batch_size": config["per_device_train_batch_size"],
                "eval_batch_size": config["per_device_eval_batch_size"],
                "best_epoch": result["best_epoch"],
                f"best_{primary_eval_metric_key}": result["best_metric"],
                "adapter_params": config.get("parameter_stats", {}).get("adapter_params"),
                "classifier_trainable_params": config.get("parameter_stats", {}).get("classifier_trainable_params"),
                "other_trainable_params": config.get("parameter_stats", {}).get("other_trainable_params"),
                "total_trainable_params": config.get("parameter_stats", {}).get("total_trainable_params"),
                "all_params": config.get("parameter_stats", {}).get("all_params"),
                "trainable_ratio": config.get("parameter_stats", {}).get("trainable_ratio"),
                "initial_train_loss": initial_train_metrics.get("train_init_loss"),
                "initial_eval_loss": initial_eval_metrics.get("eval_init_loss"),
                f"initial_{primary_eval_metric_key}": initial_eval_metrics.get(initial_eval_metric_key),
                "final_eval_loss": final_metrics.get("eval_loss"),
                primary_eval_metric_key: final_metrics.get(primary_eval_metric_key),
                "experiment_dir": result["experiment_dir"],
            }
        )
    return rows
