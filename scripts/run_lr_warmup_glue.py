"""Run six encoder/classifier LR pairs per task with a common 0.06 warmup."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODER_LEARNING_RATES = [1e-3, 2e-3]
CLASSIFIER_LEARNING_RATES = [1e-4, 2e-4, 5e-4]
LEARNING_RATE_PAIRS = [(encoder, head) for encoder in ENCODER_LEARNING_RATES
                      for head in CLASSIFIER_LEARNING_RATES]


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_state(gpu):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    values = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return dict(zip(("gpu", "total_mib", "used_mib", "free_mib", "utilization"), values))


def wait_for_memory(gpu, minimum):
    while True:
        state = gpu_state(gpu)
        if state["free_mib"] >= minimum:
            print(f"GPU memory check: {state}", flush=True)
            return state
        print(f"Waiting for {minimum} MiB free on GPU {gpu}: {state}", flush=True)
        time.sleep(30)


def build_plan(args):
    model_path = Path(args.model_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    cache = Path(args.decomposition_cache_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    plans = []
    for task in ("cola", "mrpc"):
        source = PROJECT_ROOT / "examples" / f"sweep_lr_warmup_{task}_e15.json"
        specs = json.loads(source.read_text(encoding="utf-8"))
        pairs = [(s["learning_rate"], s["classifier_learning_rate"]) for s in specs]
        if pairs != LEARNING_RATE_PAIRS:
            raise ValueError(f"Expected all six encoder/classifier learning-rate pairs in {source}.")
        for spec in specs:
            if (spec["glue_task"] != task or spec["num_train_epochs"] != 15
                    or spec["attn_ranks"] != [24, 4, 16, 32]
                    or spec["ffn_ranks"] != [2, 4, 64, 64]
                    or spec["warmup_ratio"] != 0.06 or spec["warmup_steps"] != 0
                    or spec["lr_scheduler_type"] != "linear"):
                raise ValueError(f"Unexpected task, ranks, epochs, or scheduler in {source}.")
        config_path = output / f"{task}_config.json"
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "train_astra_core.py"),
            "--model-path", str(model_path),
            "--dataset-path", str(dataset_root / f"glue_{task}"),
            "--glue-task", task,
            "--decomposition-cache-dir", str(cache),
            "--sweep-config", str(config_path),
            "--sweep-output-dir", str(output / task),
            "--gpu-memory-fraction", str(args.gpu_memory_fraction),
        ]
        plans.append({"task": task, "runs": len(specs), "epochs": 15, "specs": specs,
                      "config_path": str(config_path), "log_path": str(output / f"{task}.log"),
                      "command": command})
    return model_path, dataset_root, cache, output, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--decomposition-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.48)
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or not 0 < args.gpu_memory_fraction <= 1 or args.min_free_mib <= 0:
        parser.error("Invalid GPU, memory fraction, or memory threshold.")
    model_path, dataset_root, cache, output, plans = build_plan(args)
    manifest = {
        "gpu": args.gpu, "gpu_memory_fraction": args.gpu_memory_fraction,
        "min_free_mib": args.min_free_mib, "total_runs": sum(p["runs"] for p in plans),
        "model_path": str(model_path), "dataset_root": str(dataset_root),
        "decomposition_cache_dir": str(cache), "code_snapshot": str(PROJECT_ROOT),
        "tasks": plans,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path}")
    for task in ("cola", "mrpc"):
        if not (dataset_root / f"glue_{task}" / "dataset_dict.json").is_file():
            raise FileNotFoundError(f"Missing saved {task} dataset.")
    for family in ("q", "k", "v", "o", "ffn"):
        ranks = "2-4-64-64" if family == "ffn" else "24-4-16-32"
        if not (cache / f"{family}_r{ranks}.pt").is_file():
            raise FileNotFoundError(f"Missing decomposition cache for {family}: {cache}")
    # A deployment may already have placed its code snapshot in this directory.
    # Refuse reuse of an output directory that has any previous batch artifacts.
    reserved = ("batch_manifest.json", "status.json", ".launch.lock",
                "cola_config.json", "mrpc_config.json", "cola.log", "mrpc.log", "cola", "mrpc")
    if any((output / name).exists() for name in reserved):
        raise FileExistsError(f"This batch output has already been used: {output}")
    initial_gpu = gpu_state(args.gpu)
    if initial_gpu["free_mib"] < args.min_free_mib:
        raise RuntimeError(f"Insufficient free GPU memory at launch: {initial_gpu}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launch.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    manifest["gpu_at_start"] = initial_gpu
    write_json(output / "batch_manifest.json", manifest)
    for plan in plans:
        write_json(Path(plan["config_path"]), plan["specs"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    status = {"state": "running", "launcher_pid": os.getpid(), "completed_tasks": []}
    try:
        for plan in plans:
            status["current_task"] = plan["task"]
            status["state"] = "waiting_for_memory"
            write_json(output / "status.json", status)
            wait_for_memory(args.gpu, args.min_free_mib)
            status["state"] = "running"
            write_json(output / "status.json", status)
            print(f"Starting {plan['task']}: six encoder/classifier LR pairs, 15 epochs, warmup_ratio=0.06; "
                  f"log={plan['log_path']}", flush=True)
            with Path(plan["log_path"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(plan["command"], cwd=Path.cwd(), env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status["training_pid"] = process.pid
                write_json(output / "status.json", status)
                code = process.wait()
            if code:
                raise RuntimeError(f"{plan['task']} exited with code {code}; see {plan['log_path']}")
            status["completed_tasks"].append(plan["task"])
            write_json(output / "status.json", status)
            print(f"Completed {plan['task']}.", flush=True)
        status.update(state="complete", current_task=None, training_pid=None)
        write_json(output / "status.json", status)
        print(f"All {manifest['total_runs']} runs completed. Results: {output}", flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
