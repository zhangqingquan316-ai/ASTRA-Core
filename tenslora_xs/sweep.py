"""Naming, sweep parsing, and shared experiment bookkeeping."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .constants import SWEEP_KEY_ALIASES
from .tasks import infer_families_from_modules


def resolve_target_families(args):
    """Resolve the actual family list after handling backward-compatible aliases."""
    if args.target_modules:
        return infer_families_from_modules(args.target_modules)
    return list(args.target_families)


def sanitize_experiment_name(name):
    """Keep run names filesystem-safe without making them unreadable."""
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "experiment"


def build_default_experiment_name(spec, index):
    """Generate a compact default name when a sweep item does not provide one."""
    parts = [f"exp_{index:02d}"]
    if "tuning_mode" in spec:
        parts.append(spec["tuning_mode"])
    if "attn_ranks" in spec:
        parts.append("r" + "-".join(str(x) for x in spec["attn_ranks"]))
    if "attn_alpha" in spec:
        parts.append(f"a{spec['attn_alpha']}")
    if "multiplicative_num_bases" in spec and spec.get("tuning_mode") in {"multiplicative", "both"}:
        parts.append(f"mb{spec['multiplicative_num_bases']}")
    if "num_train_epochs" in spec:
        parts.append(f"e{spec['num_train_epochs']}")
    return sanitize_experiment_name("_".join(parts))


def normalize_sweep_spec(spec):
    """Translate CLI-style sweep keys into argparse-style attribute names."""
    normalized = {}
    for key, value in spec.items():
        normalized[SWEEP_KEY_ALIASES.get(key, key)] = value
    return normalized


def get_model_tag(model_path):
    """Turn the local model folder name into a safe cache subdirectory name."""
    name = Path(model_path).name or "model"
    return sanitize_experiment_name(name.replace("-", "_"))


def load_sweep_experiments(args):
    """Expand CLI arguments into one or many experiment namespaces."""
    base_args = deepcopy(vars(args))

    if args.sweep_config is None:
        single_args = deepcopy(base_args)
        single_args["run_name"] = sanitize_experiment_name(args.run_name or f"{args.glue_task}_run")
        single_args["single_output_mode"] = args.run_name is None
        return [argparse.Namespace(**single_args)], False

    sweep_path = Path(args.sweep_config)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {sweep_path.resolve()}")

    experiments = json.loads(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Sweep config must be a non-empty JSON list.")

    protected_keys = {"sweep_config", "sweep_output_dir", "run_name", "single_output_mode"}

    experiment_args_list = []
    seen_names = set()
    for index, raw_spec in enumerate(experiments, start=1):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Sweep experiment at index {index} must be a JSON object.")

        spec = normalize_sweep_spec(raw_spec)
        merged = deepcopy(base_args)
        for key, value in spec.items():
            if key == "name":
                continue
            if key in protected_keys:
                raise ValueError(f"Field '{key}' cannot be overridden inside sweep config.")
            if key not in merged:
                raise ValueError(f"Unknown sweep field '{key}' in experiment {index}.")
            merged[key] = value

        experiment_name = sanitize_experiment_name(
            spec.get("name") or build_default_experiment_name(spec, index)
        )
        if experiment_name in seen_names:
            raise ValueError(f"Duplicate experiment name '{experiment_name}' in sweep config.")
        seen_names.add(experiment_name)

        merged["run_name"] = experiment_name
        merged["single_output_mode"] = False
        experiment_args_list.append(argparse.Namespace(**merged))

    task_names = {exp.glue_task for exp in experiment_args_list}
    if len(task_names) != 1:
        raise ValueError("All experiments in one sweep must use the same GLUE task.")

    return experiment_args_list, True
