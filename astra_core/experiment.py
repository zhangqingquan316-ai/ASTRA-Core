"""High-level experiment workflow for ASTRA-Core training."""

import gc
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .artifacts import (
    create_sweep_summary_rows,
    extract_history_records,
    prepend_initial_history,
    save_history_artifacts,
    save_json,
    save_sweep_comparison_plot,
    save_trainable_checkpoint,
    write_csv_rows,
)
from .modeling import (
    build_runtime_family_from_decomposition_cache,
    get_attention_geometry,
    get_ffn_geometry,
    get_or_create_decomposition_cache_entry,
    patch_model_with_tucker,
    unfreeze_classifier_head,
)
from .sweep import get_model_tag, load_sweep_experiments, resolve_target_families
from .tasks import build_compute_metrics, get_primary_eval_metric_key, get_task_config, validate_local_inputs
from .training_state import BestModelByMetricCallback, load_trainable_state


def cleanup_experiment_memory():
    """Release Python and CUDA memory between experiments in a sweep."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_experiment_dirs(args, sweep_root=None):
    """Build the output-directory layout for a single experiment."""
    if sweep_root is not None:
        experiment_root = Path(sweep_root) / args.run_name
        results_dir = experiment_root / "results"
        final_dir = experiment_root / "final"
        return experiment_root, results_dir, final_dir

    if getattr(args, "single_output_mode", True):
        results_dir = Path(f"./astra_core_{args.glue_task}_results")
        final_dir = Path(f"./astra_core_{args.glue_task}_final")
        experiment_root = final_dir
    else:
        experiment_root = Path("./astra_core_runs") / args.glue_task / args.run_name
        results_dir = experiment_root / "results"
        final_dir = experiment_root / "final"

    return experiment_root, results_dir, final_dir


def print_training_configuration(args, target_families, experiment_name):
    """Print the key hyperparameters so each run is easy to audit."""
    print(f"Experiment name: {experiment_name}")
    print("Current training configuration:")
    print(f"  target_families = {target_families}")
    print(f"  tuning_mode = {args.tuning_mode}")
    print(f"  attn_ranks = {tuple(args.attn_ranks)}")
    print(f"  ffn_ranks = {tuple(args.ffn_ranks)}")
    print(f"  attn_alpha = {args.attn_alpha}")
    print(f"  ffn_alpha = {args.ffn_alpha}")
    print(f"  multiplicative_num_bases = {args.multiplicative_num_bases}")
    print(f"  num_train_epochs = {args.num_train_epochs}")
    print(f"  learning_rate = {args.learning_rate}")
    print(f"  per_device_train_batch_size = {args.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size = {args.per_device_eval_batch_size}")
    print(f"  weight_decay = {args.weight_decay}")
    print(f"  seed = {args.seed}")


def load_local_model_and_tokenizer(args, task_config):
    """Load the local tokenizer and classification model for one experiment."""
    print("Loading local model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    model_kwargs = {
        "num_labels": task_config["num_labels"],
        "low_cpu_mem_usage": False,
        "local_files_only": True,
    }
    if task_config["is_regression"]:
        model_kwargs["problem_type"] = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **model_kwargs)
    return tokenizer, model


def prepare_tokenized_datasets(args, tokenizer, task_config, resolved_dataset_path):
    """Load the local dataset and tokenize the expected sentence fields."""
    print("Loading and preparing local dataset...")
    dataset = load_from_disk(resolved_dataset_path)
    if "train" not in dataset or "validation" not in dataset:
        raise KeyError(
            f"Dataset must contain at least train and validation splits. "
            f"Found: {list(dataset.keys())}"
        )

    sentence1_key, sentence2_key = task_config["text_keys"]
    train_columns = dataset["train"].column_names
    missing_columns = [key for key in (sentence1_key, sentence2_key) if key and key not in train_columns]
    if missing_columns:
        raise KeyError(
            f"Task '{args.glue_task}' expects columns {missing_columns}, "
            f"but train split has columns: {train_columns}"
        )

    def tokenize_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                max_length=args.max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return dataset, tokenized_datasets, data_collator


def build_training_arguments(args, results_dir):
    """Create Hugging Face TrainingArguments with sensible dtype selection."""
    bf16_supported = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_bf16 = bf16_supported
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=str(results_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        save_safetensors=False,
        report_to="none",
        seed=args.seed,
    )


def compute_parameter_statistics(model):
    """Compute the parameter-count summary used for logging and result files.

    `adapter_params` is intended to answer the paper-facing question
    "how many trainable parameters are introduced by the ASTRA adapter itself?".
    It therefore counts only trainable parameters that live inside
    `astra_core_shared_state_*` modules, and excludes:
    - the classifier head (`classifier.*` / `score.*`)
    - frozen Tucker factors / cached decomposition tensors stored as buffers
    """
    named_parameters = list(model.named_parameters())
    adapter_params = sum(
        param.numel()
        for name, param in named_parameters
        if param.requires_grad and name.startswith("astra_core_shared_state_")
    )
    classifier_trainable_params = sum(
        param.numel()
        for name, param in named_parameters
        if param.requires_grad and (name.startswith("classifier.") or name.startswith("score."))
    )
    total_trainable_params = sum(param.numel() for _, param in named_parameters if param.requires_grad)
    all_params = sum(param.numel() for _, param in named_parameters)
    other_trainable_params = total_trainable_params - adapter_params - classifier_trainable_params
    trainable_ratio = float(total_trainable_params / all_params) if all_params else 0.0

    return {
        "adapter_params": int(adapter_params),
        "classifier_trainable_params": int(classifier_trainable_params),
        "other_trainable_params": int(other_trainable_params),
        "total_trainable_params": int(total_trainable_params),
        "all_params": int(all_params),
        "trainable_ratio": trainable_ratio,
    }


def run_single_experiment(
    args,
    experiment_index,
    total_experiments,
    sweep_root=None,
    decomposition_cache_entries=None,
):
    """Run one fully specified experiment from start to finish."""
    set_seed(args.seed)

    experiment_root, results_dir, final_dir = prepare_experiment_dirs(args, sweep_root=sweep_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset_path = args.dataset_path or f"./local_datasets/glue_{args.glue_task}"
    validate_local_inputs(args.model_path, resolved_dataset_path)

    target_families = resolve_target_families(args)
    task_config = get_task_config(args.glue_task)
    primary_eval_metric_key = get_primary_eval_metric_key(args.glue_task)

    if args.tuning_mode in {"multiplicative", "both"} and args.multiplicative_num_bases <= 0:
        raise ValueError(
            "When tuning_mode is 'multiplicative' or 'both', multiplicative_num_bases must be positive. "
            f"Got {args.multiplicative_num_bases}."
        )

    print("=" * 80)
    print(f"Running experiment {experiment_index}/{total_experiments}: {args.run_name}")
    print(f"GLUE task: {args.glue_task}")
    print(f"Experiment directory: {experiment_root}")
    print_training_configuration(args, target_families, args.run_name)

    tokenizer, model = load_local_model_and_tokenizer(args, task_config)

    num_heads, head_dim, hidden_size = get_attention_geometry(model)
    ffn_m, ffn_n = get_ffn_geometry(model)
    print(
        f"Attention geometry: num_heads={num_heads}, head_dim={head_dim}, hidden_size={hidden_size}"
    )
    print(f"FFN canonical geometry target: m={ffn_m}, n={ffn_n}")

    for param in model.parameters():
        param.requires_grad = False

    if decomposition_cache_entries is None:
        decomposition_cache_entries = {}
    if not getattr(args, "_resolved_decomposition_cache_dir", None):
        raise ValueError("Decomposition cache directory was not prepared.")

    decomposition_results = []
    decomposition_cache_root = Path(args._resolved_decomposition_cache_dir)
    print("Preparing Tucker/HOOI decompositions for requested families...")
    for family_name in target_families:
        requested_ranks = tuple(args.ffn_ranks) if family_name == "ffn" else tuple(args.attn_ranks)
        alpha = args.ffn_alpha if family_name == "ffn" else args.attn_alpha
        cache_entry = get_or_create_decomposition_cache_entry(
            model=model,
            model_path=args.model_path,
            cache_root=decomposition_cache_root,
            family_name=family_name,
            ranks=requested_ranks,
            num_heads=num_heads,
            shared_cache_entries=decomposition_cache_entries,
        )
        print(
            f"[{family_name}] decomposition method = {cache_entry['method']}, "
            f"cached ranks = {tuple(cache_entry['ranks'])}"
        )
        result = build_runtime_family_from_decomposition_cache(
            cache_entry=cache_entry,
            alpha=alpha,
            tuning_mode=args.tuning_mode,
            multiplicative_num_bases=args.multiplicative_num_bases,
            base_seed=args.seed,
        )
        decomposition_results.append(result)

    print("Patching model with Tucker adapters...")
    for result in decomposition_results:
        model = patch_model_with_tucker(model, result)

    unfreeze_classifier_head(model)

    parameter_stats = compute_parameter_statistics(model)
    print(
        f"Adapter params: {parameter_stats['adapter_params']:,d} || "
        f"Classifier trainable params: {parameter_stats['classifier_trainable_params']:,d} || "
        f"Total trainable params: {parameter_stats['total_trainable_params']:,d} || "
        f"All params: {parameter_stats['all_params']:,d} || "
        f"Percentage: {100 * parameter_stats['trainable_ratio']:.4f}%"
    )

    dataset, tokenized_datasets, data_collator = prepare_tokenized_datasets(
        args=args,
        tokenizer=tokenizer,
        task_config=task_config,
        resolved_dataset_path=resolved_dataset_path,
    )
    compute_metrics = build_compute_metrics(args.glue_task)
    training_args = build_training_arguments(args, results_dir)
    best_callback = BestModelByMetricCallback(metric_name=primary_eval_metric_key)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[best_callback],
    )

    print("Computing initial metrics before training...")
    initial_train_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["train"],
        metric_key_prefix="train_init",
    )
    initial_eval_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        metric_key_prefix="eval_init",
    )
    trainer.state.log_history.clear()

    print("Starting training...")
    trainer.train()
    plot_log_history = list(trainer.state.log_history)

    if best_callback.best_state is not None:
        print(
            f"Restoring best model from epoch={best_callback.best_epoch} "
            f"with {primary_eval_metric_key}={best_callback.best_metric:.6f}"
        )
        load_trainable_state(model, best_callback.best_state)
    else:
        print("Warning: best model state was not captured. Using final training state.")

    print("Evaluating best model...")
    final_metrics = trainer.evaluate()
    print(final_metrics)

    raw_train_history, raw_eval_history = extract_history_records(
        plot_log_history,
        primary_eval_metric_key=primary_eval_metric_key,
    )
    train_history, eval_history = prepend_initial_history(
        raw_train_history,
        raw_eval_history,
        initial_train_metrics=initial_train_metrics,
        initial_eval_metrics=initial_eval_metrics,
        task_name=args.glue_task,
    )

    save_trainable_checkpoint(
        model=model,
        output_dir=final_dir,
        args=args,
        target_families=target_families,
        experiment_name=args.run_name,
        resolved_dataset_path=resolved_dataset_path,
        parameter_stats=parameter_stats,
    )
    save_history_artifacts(
        output_dir=experiment_root,
        train_history=train_history,
        eval_history=eval_history,
        initial_train_metrics=initial_train_metrics,
        initial_eval_metrics=initial_eval_metrics,
        final_metrics=final_metrics,
        task_name=args.glue_task,
        primary_eval_metric_key=primary_eval_metric_key,
        experiment_name=args.run_name,
    )

    experiment_config = {
        "experiment_name": args.run_name,
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

    summary_payload = {
        "experiment_name": args.run_name,
        "best_epoch": best_callback.best_epoch,
        "best_metric": best_callback.best_metric,
        "primary_eval_metric_key": primary_eval_metric_key,
        "initial_train_metrics": initial_train_metrics,
        "initial_eval_metrics": initial_eval_metrics,
        "final_metrics": final_metrics,
        "experiment_dir": str(experiment_root.resolve()),
        "final_dir": str(final_dir.resolve()),
        "parameter_stats": parameter_stats,
    }

    save_json(experiment_root / "experiment_config.json", experiment_config)
    save_json(experiment_root / "experiment_summary.json", summary_payload)

    result = {
        "experiment_name": args.run_name,
        "experiment_dir": str(experiment_root.resolve()),
        "final_dir": str(final_dir.resolve()),
        "config": experiment_config,
        "initial_train_metrics": initial_train_metrics,
        "initial_eval_metrics": initial_eval_metrics,
        "final_metrics": final_metrics,
        "best_epoch": best_callback.best_epoch,
        "best_metric": best_callback.best_metric,
        "eval_history": eval_history,
        "parameter_stats": parameter_stats,
    }

    del trainer
    del model
    del tokenized_datasets
    del dataset
    cleanup_experiment_memory()

    print(f"Finished experiment: {args.run_name}")
    return result


def run_from_cli(args):
    """Run either one experiment or a sweep from parsed CLI arguments."""
    experiment_args_list, is_sweep = load_sweep_experiments(args)

    if is_sweep:
        if args.sweep_output_dir:
            sweep_root = Path(args.sweep_output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_root = Path("./astra_core_sweeps") / f"{experiment_args_list[0].glue_task}_{timestamp}"
        sweep_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep output directory: {sweep_root}")
    else:
        sweep_root = None

    if args.decomposition_cache_dir:
        decomposition_cache_root = Path(args.decomposition_cache_dir)
    else:
        decomposition_cache_root = Path("./tucker_hooi_cache") / get_model_tag(experiment_args_list[0].model_path)
    decomposition_cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Tucker/HOOI decomposition cache directory: {decomposition_cache_root}")

    for experiment_args in experiment_args_list:
        experiment_args._resolved_decomposition_cache_dir = str(decomposition_cache_root.resolve())

    decomposition_cache_entries = {}
    experiment_results = []
    total_experiments = len(experiment_args_list)
    for index, experiment_args in enumerate(experiment_args_list, start=1):
        experiment_results.append(
            run_single_experiment(
                args=experiment_args,
                experiment_index=index,
                total_experiments=total_experiments,
                sweep_root=sweep_root,
                decomposition_cache_entries=decomposition_cache_entries,
            )
        )

    if is_sweep:
        task_name = experiment_args_list[0].glue_task
        primary_eval_metric_key = get_primary_eval_metric_key(task_name)
        summary_rows = create_sweep_summary_rows(experiment_results, task_name)

        write_csv_rows(
            sweep_root / "sweep_summary.csv",
            summary_rows,
            preferred_fields=[
                "experiment_name",
                "attn_ranks",
                "attn_alpha",
                "num_train_epochs",
                "best_epoch",
                f"best_{primary_eval_metric_key}",
                primary_eval_metric_key,
            ],
        )
        save_json(sweep_root / "sweep_summary.json", summary_rows)
        save_sweep_comparison_plot(
            experiment_results=experiment_results,
            output_dir=sweep_root,
            task_name=task_name,
            primary_eval_metric_key=primary_eval_metric_key,
        )
        print(f"Sweep completed. Summary saved to {sweep_root}")
    else:
        print("Single experiment completed.")
