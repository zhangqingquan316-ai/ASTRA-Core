"""Run the existing 34 rank settings on CoLA/MRPC/SST-2, in sequence."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_EPOCHS = (("cola", 15), ("mrpc", 15), ("sst2", 20))


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.getenv("TUCKER_MODEL_PATH", "./local_models/roberta-large"),
        help="Existing local RoBERTa-large model directory.",
    )
    parser.add_argument("--dataset-root", default="./local_datasets")
    parser.add_argument("--output-dir", help="A new directory for this batch of 102 runs.")
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index to use.")
    parser.add_argument(
        "--prepare-missing-datasets", action="store_true",
        help="Download and save missing GLUE datasets before training; existing directories are preserved.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without starting training or writing files."
    )
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative GPU index.")
    return args


def build_plan(args):
    model_path = project_path(args.model_path)
    dataset_root = project_path(args.dataset_root)
    output_root = project_path(
        args.output_dir
        or f"./astra_core_sweeps/ofat_glue_{datetime.now():%Y%m%d_%H%M%S}"
    )
    plans = []
    original_ranks = None
    for task, epochs in TASK_EPOCHS:
        source = PROJECT_ROOT / "examples" / f"sweep_ofat_{task}_e{epochs}.json"
        specs = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(specs, list) or len(specs) != 34:
            raise ValueError(f"Expected 34 rank settings in {source}.")
        ranks = []
        names = set()
        for spec in specs:
            if spec["glue_task"] != task or spec["num_train_epochs"] != epochs:
                raise ValueError(f"Task/epoch mismatch in {source}.")
            for field in ("attn_ranks", "ffn_ranks"):
                values = spec[field]
                if len(values) != 4 or any(type(v) is not int or v <= 0 for v in values):
                    raise ValueError(f"Invalid {field} in {source}.")
            if spec["name"] in names:
                raise ValueError(f"Duplicate experiment name in {source}.")
            names.add(spec["name"])
            ranks.append((tuple(spec["attn_ranks"]), tuple(spec["ffn_ranks"])))
        if len(set(ranks)) != 34 or (original_ranks is not None and ranks != original_ranks):
            raise ValueError("All three tasks must use the same 34 distinct rank pairs in the same order.")
        original_ranks = ranks
        config_path = output_root / f"{task}_config.json"
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "train_astra_core.py"),
            "--model-path", str(model_path),
            "--dataset-path", str(dataset_root / f"glue_{task}"),
            "--glue-task", task,
            "--sweep-config", str(config_path),
            "--sweep-output-dir", str(output_root / task),
        ]
        plans.append({
            "task": task, "epochs": epochs, "specs": specs,
            "source_config": str(source), "config_path": str(config_path),
            "dataset_path": str(dataset_root / f"glue_{task}"),
            "log_path": str(output_root / f"{task}.log"), "command": command,
        })
    return model_path, output_root, plans


def main():
    args = parse_args()
    model_path, output_root, plans = build_plan(args)
    if output_root.exists():
        raise ValueError(f"Output directory already exists; choose a new --output-dir: {output_root}")
    print(f"GPU: {args.gpu}; 34 rank settings x 3 tasks = 102 new training runs.", flush=True)
    print(f"Output directory: {output_root}", flush=True)
    for plan in plans:
        print(f"{plan['task']}: 34 runs, {plan['epochs']} epochs each", flush=True)
        print(shlex.join(plan["command"]), flush=True)
    if args.dry_run:
        return

    # Check all inputs before spending GPU time on the first task.
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Local model config not found: {model_path / 'config.json'}")
    for plan in plans:
        dataset_path = Path(plan["dataset_path"])
        if not (dataset_path / "dataset_dict.json").is_file():
            if args.prepare_missing_datasets and not dataset_path.exists():
                print(f"Preparing missing dataset: {plan['task']}", flush=True)
                subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_glue_dataset.py"),
                     "--task", plan["task"], "--output-dir", str(dataset_path)],
                    cwd=PROJECT_ROOT, check=True,
                )
            if (dataset_path / "dataset_dict.json").is_file():
                continue
            raise FileNotFoundError(
                f"Saved GLUE dataset not found: {dataset_path}. Prepare it first with: "
                f"python scripts/prepare_glue_dataset.py --task {plan['task']} "
                f"--output-dir {shlex.quote(str(dataset_path))}"
            )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    for plan in plans:
        Path(plan["config_path"]).write_text(
            json.dumps(plan["specs"], indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "gpu": args.gpu,
        "model_path": str(model_path),
        "total_runs": sum(len(plan["specs"]) for plan in plans),
        "tasks": [{key: value for key, value in plan.items() if key != "specs"} for plan in plans],
    }
    (output_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for plan in plans:
        print(f"Starting {plan['task']}; training log: {plan['log_path']}", flush=True)
        with Path(plan["log_path"]).open("x", encoding="utf-8") as log:
            result = subprocess.run(
                plan["command"], cwd=PROJECT_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(
                f"{plan['task']} failed with exit code {result.returncode}; see {plan['log_path']}"
            )
        print(f"Finished {plan['task']}.", flush=True)
    print(f"All 102 runs completed. Results: {output_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
