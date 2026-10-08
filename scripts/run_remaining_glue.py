"""Run SST-2, QNLI, RTE and STS-B at fixed encoder/head rates for 20 epochs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_lr_warmup_glue import gpu_state, wait_for_memory, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sst2", "qnli", "rte", "stsb")


def build_plan(args):
    model = Path(args.model_path).expanduser().resolve()
    datasets = Path(args.dataset_root).expanduser().resolve()
    cache = Path(args.decomposition_cache_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    specs = json.loads((PROJECT_ROOT / "examples/remaining_glue_e20.json").read_text(encoding="utf-8"))
    if [spec["glue_task"] for spec in specs] != list(TASKS):
        raise ValueError("Expected exactly SST-2, QNLI, RTE, STS-B in this order.")
    plans = []
    for spec in specs:
        task = spec["glue_task"]
        expected = dict(learning_rate=.002, classifier_learning_rate=.0001,
                        warmup_ratio=.06, warmup_steps=0, num_train_epochs=20,
                        attn_ranks=[24, 4, 16, 32], ffn_ranks=[2, 4, 64, 64],
                        target_families=["q", "k", "v", "o", "ffn"], tuning_mode="manifold",
                        manifold_policy="adaptive", lr_scheduler_type="linear", seed=42,
                        per_device_train_batch_size=32, per_device_eval_batch_size=32,
                        gradient_accumulation_steps=1, max_length=128, weight_decay=.01)
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected configuration for {task}.")
        config = output / f"{task}_config.json"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "train_astra_core.py"),
                   "--model-path", str(model), "--dataset-path", str(datasets / f"glue_{task}"),
                   "--glue-task", task, "--decomposition-cache-dir", str(cache),
                   "--sweep-config", str(config), "--sweep-output-dir", str(output / task),
                   "--gpu-memory-fraction", str(args.gpu_memory_fraction)]
        plans.append(dict(task=task, runs=1, epochs=20, specs=[spec], config_path=str(config),
                          log_path=str(output / f"{task}.log"), command=command))
    return model, datasets, cache, output, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--decomposition-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.48)
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or not 0 < args.gpu_memory_fraction <= 1 or args.min_free_mib <= 0:
        parser.error("Invalid GPU, memory fraction, or memory threshold.")
    model, datasets, cache, output, plans = build_plan(args)
    manifest = dict(gpu=args.gpu, gpu_memory_fraction=args.gpu_memory_fraction,
                    min_free_mib=args.min_free_mib, total_runs=len(plans), tasks=plans,
                    model_path=str(model), dataset_root=str(datasets),
                    decomposition_cache_dir=str(cache), code_snapshot=str(PROJECT_ROOT))
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model}")
    for task in TASKS:
        if not (datasets / f"glue_{task}" / "dataset_dict.json").is_file():
            raise FileNotFoundError(f"Missing saved dataset: {task}")
    for family in ("q", "k", "v", "o", "ffn"):
        ranks = "2-4-64-64" if family == "ffn" else "24-4-16-32"
        if not (cache / f"{family}_r{ranks}.pt").is_file():
            raise FileNotFoundError(f"Missing cached decomposition: {family}")
    reserved = ["batch_manifest.json", "status.json", ".launch.lock"]
    reserved += [name for task in TASKS for name in (task, f"{task}.log", f"{task}_config.json")]
    if any((output / name).exists() for name in reserved):
        raise FileExistsError(f"This batch output has already been used: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launch.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    manifest["gpu_at_submission"] = gpu_state(args.gpu)
    write_json(output / "batch_manifest.json", manifest)
    for plan in plans:
        write_json(Path(plan["config_path"]), plan["specs"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    status = dict(state="submitted", launcher_pid=os.getpid(), completed_tasks=[])
    try:
        for plan in plans:
            status.update(state="waiting_for_memory", current_task=plan["task"], training_pid=None)
            write_json(output / "status.json", status)
            wait_for_memory(args.gpu, args.min_free_mib)
            print(f"Starting {plan['task']}: encoder_lr=.002, head_lr=.0001, "
                  f"warmup_ratio=.06, epochs=20; log={plan['log_path']}", flush=True)
            with Path(plan["log_path"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(plan["command"], cwd=Path.cwd(), env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status.update(state="running", training_pid=process.pid)
                write_json(output / "status.json", status)
                code = process.wait()
            if code:
                raise RuntimeError(f"{plan['task']} exited with code {code}; see {plan['log_path']}")
            status["completed_tasks"].append(plan["task"])
            status.update(training_pid=None)
            write_json(output / "status.json", status)
            print(f"Completed {plan['task']}.", flush=True)
        status.update(state="complete", current_task=None, training_pid=None)
        write_json(output / "status.json", status)
        print(f"All four tasks completed. Results: {output}", flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
