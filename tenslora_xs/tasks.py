"""Task definitions, input validation, and evaluation metrics."""

from pathlib import Path

import numpy as np

from .constants import MODULE_TO_FAMILY, TASK_CONFIGS


def validate_local_inputs(model_path, dataset_path):
    """Ensure both the local model folder and dataset folder exist."""
    model_dir = Path(model_path)
    dataset_dir = Path(dataset_path)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local model directory does not exist: {model_dir.resolve()}. "
            "Please point --model-path to the real Hugging Face model folder."
        )
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset directory does not exist: {dataset_dir.resolve()}. "
            "Please point --dataset-path to the real datasets.save_to_disk folder."
        )


def get_task_config(task_name):
    """Return the GLUE task metadata used for tokenization and evaluation."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unsupported GLUE task '{task_name}'. Supported tasks: {sorted(TASK_CONFIGS)}"
        )
    return TASK_CONFIGS[task_name]


def get_primary_eval_metric_key(task_name):
    """Return the key used by Hugging Face Trainer for the main eval metric."""
    return f"eval_{get_task_config(task_name)['metric']}"


def get_prefixed_metric_key(task_name, prefix):
    """Return the metric key when the Trainer uses a custom prefix."""
    return f"{prefix}_{get_task_config(task_name)['metric']}"


def infer_families_from_modules(target_modules):
    """Map user-facing module aliases to the internal family names."""
    families = []
    for module_name in target_modules:
        family = MODULE_TO_FAMILY.get(module_name.lower())
        if family is None:
            raise ValueError(
                f"Unsupported target module '{module_name}'. "
                f"Supported values include: {sorted(MODULE_TO_FAMILY)}"
            )
        if family not in families:
            families.append(family)
    return families


def compute_binary_accuracy(predictions, labels):
    return float((predictions == labels).mean())


def compute_matthews_correlation(predictions, labels):
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0

    return float((tp * tn - fp * fn) / np.sqrt(denominator))


def compute_pearson_correlation(predictions, labels):
    predictions = predictions.astype(np.float64)
    labels = labels.astype(np.float64)
    pred_std = predictions.std()
    label_std = labels.std()
    if pred_std == 0 or label_std == 0:
        return 0.0
    return float(np.corrcoef(predictions, labels)[0, 1])


def build_compute_metrics(task_name):
    """Build the Trainer callback used to compute task-specific metrics."""
    task_config = get_task_config(task_name)
    metric_name = task_config["metric"]
    is_regression = task_config["is_regression"]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.squeeze(logits) if is_regression else np.argmax(logits, axis=-1)

        if metric_name == "accuracy":
            return {"accuracy": compute_binary_accuracy(predictions, labels)}
        if metric_name == "matthews":
            return {"matthews": compute_matthews_correlation(predictions, labels)}
        if metric_name == "pearson":
            return {"pearson": compute_pearson_correlation(predictions, labels)}
        raise ValueError(f"Unsupported metric '{metric_name}' for task '{task_name}'.")

    return compute_metrics

